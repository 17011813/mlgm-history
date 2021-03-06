"""Simple MAML implementation.

Based on algorithm 1 from:
Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning
for fast adaptation of deep networks." Proceedings of the 34th International
Conference on Machine Learning-Volume 70. JMLR. org, 2017.

https://arxiv.org/pdf/1703.03400.pdf
"""
import numpy as np
import tensorflow as tf
from mlgm.logger import Logger
from mlgm.utils import gen_fig

class Maml:
    def __init__(self,
                 model,
                 metasampler,
                 sess,
                 logger,
                 compute_acc=True,
                 num_updates=1,
                 update_lr=0.0001,
                 meta_lr=0.0001):
        self._model = model
        self._metasampler = metasampler
        self._sess = sess
        self._compute_acc = compute_acc
        self._num_updates = num_updates        
        self._update_lr = update_lr
        self._meta_lr = meta_lr
        self._logger = logger 
        self._init_variables()
        self._build()        
        self._logger.add_graph(self._sess.graph)        

    def _build(self):        
        with self._sess.graph.as_default():
            (self._input_a, self._label_a, self._input_b,
             self._label_b) = self._metasampler.build_inputs_and_labels(self._handle)
            #print(self._input_a.shape, self._label_a.shape, self._input_b.shape, self._label_b.shape) (7, 5, 28, 28) (7, 5, 28, 28, 1) (7, 5, 28, 28) (7, 5, 28, 28, 1)
            def task_metalearn(args):
                input_a, label_a, input_b, label_b = args

                loss_a = None
                acc_a = None
                losses_b = []
                accs_b = []
                outputs_b = []
                f_w = None                            
                for i in range(self._num_updates):
                    output_b, loss, acc, loss_b, acc_b, f_w = self._build_update(
                        input_a, label_a, input_b, label_b, self._update_lr,
                        f_w)
                    if loss_a is None:                        
                        loss_a = tf.math.reduce_mean(loss)
                        acc_a = acc               
                    outputs_b.append(output_b)
                    losses_b.append(tf.math.reduce_mean(loss_b))
                    accs_b.append(acc_b)

                return outputs_b, loss_a, acc_a, losses_b, accs_b

            out_dtype = ([tf.float32] * self._num_updates, tf.float32, tf.float32,
                         [tf.float32] * self._num_updates,
                         [tf.float32] * self._num_updates)
            elems = (self._input_a, self._label_a, self._input_b,
                     self._label_b)                     
            (self._outputsb, self._loss_a, self._acc_a, self._losses_b, 
             self._accs_b) = tf.map_fn(
                        task_metalearn,
                        elems=elems,
                        dtype=out_dtype,
                        parallel_iterations=self._metasampler.meta_batch_size)

            with tf.variable_scope("metatrain", values=[self._losses_b]):
                optimizer = tf.train.AdamOptimizer(self._meta_lr) 
                grads = optimizer.compute_gradients(
                    self._losses_b[self._num_updates - 1])
                grads = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in grads]
                self._metatrain_op = optimizer.apply_gradients(grads)

    def _init_variables(self):
        self._handle = tf.placeholder(tf.string, shape=[])

    def _build_update(self,
                      input_a,
                      label_a,
                      input_b,
                      label_b,
                      update_lr,
                      fast_weights=None):
        values = [input_a, label_a, input_b, label_b, update_lr]
        loss_a = None
        loss_b = None
        with tf.variable_scope("update", values=values):
            output_a = self._model.build_forward_pass(input_a, fast_weights)
            loss_a = self._model.build_loss(label_a, output_a)
            acc_a = self._model.build_accuracy(label_a, output_a)
            grads, weights = self._model.build_gradients(loss_a, fast_weights)
            with tf.variable_scope("fast_weights", values=[weights, grads]):
                new_fast_weights = {
                    w: weights[w] - update_lr * grads[w]
                    for w in weights
                }
            output_b = self._model.build_forward_pass(input_b,
                                                      new_fast_weights)
            loss_b = self._model.build_loss(label_b, output_b)
            acc_b = self._model.build_accuracy(label_b, output_b)
        return output_b, loss_a, acc_a, loss_b, acc_b, new_fast_weights

    def _compute_metatrain_and_acc(self, handle):
        loss_a, acc_a, losses_b, accs_b, _ = self._sess.run([
            self._loss_a, self._acc_a, self._losses_b, self._accs_b,
            self._metatrain_op], feed_dict={self._handle: handle})
        return loss_a, acc_a, losses_b, accs_b

    def _compute_metatrain(self, handle):
        loss_a, losses_b, _ = self._sess.run(            
            [self._loss_a, self._losses_b, self._metatrain_op], 
            feed_dict={self._handle: handle})
        return loss_a, losses_b

    def _compute_metatest_and_acc(self, handle): 
        return self._sess.run([
            self._input_b, self._outputsb, self._loss_a, self._acc_a, self._losses_b, self._accs_b],
            feed_dict={self._handle: handle})

    def _compute_metatest(self, handle):
        return self._sess.run([
            self._input_b, self._outputsb, self._loss_a, self._losses_b],
            feed_dict={self._handle: handle})     
            
    def test(self, test_itr, restore_model_path, log_images=True):
        assert restore_model_path

        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())        
        self._model.restore_model(restore_model_path)

        _, test_handle = self._metasampler.init_iterators(self._sess)
        self._test(test_itr, test_handle, 0, log_images)

    def train(self, train_itr, test_itr, test_interval, restore_model_path, log_images=True):
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)

        train_handle, test_handle = self._metasampler.init_iterators(self._sess)
        #print("????????????~?", train_handle.shape, test_handle.shape)
        for i in range(train_itr):
            try:
                if self._compute_acc:
                    loss_a, acc_a, losses_b, accs_b = self._compute_metatrain_and_acc(train_handle)                    
                    acc_a = np.mean(acc_a)
                    accs_b = np.array(accs_b).mean(axis=1)                    
                else:
                    loss_a, losses_b = self._compute_metatrain(train_handle)                    

                loss_a = np.mean(loss_a)
                losses_b = np.array(losses_b).mean(axis=1)
                self._logger.new_summary()
                self._logger.add_value("loss_a", loss_a)
                self._logger.add_value("loss_b/update_", losses_b.tolist())
                if self._compute_acc:
                    self._logger.add_value("acc_a", acc_a)
                    self._logger.add_value("acc_b/update_", accs_b.tolist())
                self._logger.dump_summary(i)
                self._logger.save_tf_variables(self._model.get_variables(), i, self._sess)
            except tf.errors.OutOfRangeError:
                self._metasampler.restart_train_dataset(self._sess)

            if i % test_interval == 0:
                test_img_interval = test_interval * 2
                log_images = (i % test_img_interval) == 0
                self._test(test_itr, test_handle, i, log_images)                

        self._logger.close()

    def _test(self, test_itr, test_handle, global_step, log_images=True):
        self._metasampler.restart_test_dataset(self._sess)
        total_loss_a = 0
        total_losses_b = np.array([0.] * self._num_updates)                
        for j in range(test_itr):
            try: # (7, 5, 28, 28) (5, 7, 5, 28, 28, 1)     
                input_imgs, gen_imgs, loss_a, losses_b = self._compute_metatest(test_handle) 
                #print(np.array(loss_a).shape, np.array(losses_b).shape)     loss_a (7,)      losses_b (5, 7)
                loss_a = np.mean(loss_a)
                losses_b = np.array(losses_b).mean(axis=1)
                # print(np.array(loss_a).shape, np.array(losses_b).shape)    loss_a ()      losses_b (5, )
                total_loss_a += loss_a
                total_losses_b += losses_b  # total ????????? iter ????????? ?????? ?????? ????????? ????????? ?????????????
            except tf.errors.OutOfRangeError:
                self._metasampler.restart_test_dataset(self._sess)
        
        total_loss_a = total_loss_a / test_itr
        total_losses_b = total_losses_b / test_itr
        self._logger.new_summary()                
        self._logger.add_value("test_loss_a", total_loss_a)
        self._logger.add_value("test_loss_b/update_", total_losses_b.tolist())

        #for j in range(len(gen_imgs)):  # gradient update ????????? 5??? ????????? ??? update????????? ????????? gen ???????????? ????????????.     
        self._logger.add_image(gen_fig(self._sess, input_imgs, gen_imgs[-1]), 4)
        self._logger.dump_summary(global_step)


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Maml:
    def __init__(self,
                 model,         # 여기서 model은 VAE 받아온다
                 metasampler,
                 sess,
                 logger,
                 num_updates=1,
                 update_lr=0.0001,
                 meta_lr=0.0001,
                 outliers_fraction=0.07):
        self._model = model
        self._metasampler = metasampler
        self._sess = sess
        self._compute_acc = False
        self._num_updates = num_updates        
        self._update_lr = update_lr
        self._meta_lr = meta_lr
        self._logger = logger 
        self._init_variables()
        self._build()        
        self._logger.add_graph(self._sess.graph)
        self.all_losses = 268  # 흠...?     
        self.outliers_fraction = outliers_fraction   

    def _build(self):        
        with self._sess.graph.as_default():   # (7, 5, 10) (7, 5, 10, 1) (7, 5, 10) (7, 5, 10, 1) (7, 5) (7, 5)
            self._input_a, self._label_a, self._input_b, self._label_b, self._real_a, self._real_b = self._metasampler.build_inputs_and_labels(self._handle)
            def task_metalearn(args):
                input_a, label_a, input_b, label_b = args
                loss_a = None
                acc_a = None
                losses_b = []
                accs_b = []
                outputs_b = []
                test_losses = []
                f_w = None                            
                for i in range(self._num_updates):       # print(label_b.shape)  # (5,10,1)
                    output_b, loss, acc, loss_b, acc_b, f_w, test_loss = self._build_update(input_a, label_a, input_b, label_b, self._update_lr, f_w)

                    if loss_a is None:                        
                        loss_a = tf.math.reduce_mean(loss)
                        acc_a = acc               
                    outputs_b.append(output_b)
                    losses_b.append(tf.math.reduce_mean(loss_b))   # 평균냄
                    accs_b.append(acc_b)
                    test_losses.append(test_loss)

                return outputs_b, loss_a, acc_a, losses_b, accs_b, test_losses

            out_dtype = ([tf.float32] * self._num_updates, tf.float32, tf.float32,
                         [tf.float32] * self._num_updates,
                         [tf.float32] * self._num_updates, [tf.float32] * self._num_updates)  # tf.float32 차원을 붙여서 self._test_losses를 받자
            elems = (self._input_a, self._label_a, self._input_b, self._label_b)
            self._outputsb, self._loss_a, self._acc_a, self._losses_b, self._accs_b, self._test_losses = tf.map_fn(
                        task_metalearn,
                        elems=elems,
                        dtype=out_dtype,
                        parallel_iterations=self._metasampler.meta_batch_size)

            with tf.variable_scope("metatrain", values=[self._losses_b]):   # outer meta parameter는
                optimizer = tf.train.AdamOptimizer(self._meta_lr) 
                grads = optimizer.compute_gradients(self._losses_b[self._num_updates - 1])  # 맨 마지막 애로 업데이트
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
        with tf.variable_scope("update", values=values):    # update할 때 a로 하는거 보니까 a가 meta parameter
            output_a = self._model.build_forward_pass(input_a, fast_weights)
            loss_a, _ = self._model.build_loss(label_a, output_a)
            acc_a = self._model.build_accuracy(label_a, output_a)
            grads, weights = self._model.build_gradients(loss_a, fast_weights)   # a만 gradient build 하는거 보니까 meta 맞네
            with tf.variable_scope("fast_weights", values=[weights, grads]):
                new_fast_weights = { w: weights[w] - update_lr * grads[w] for w in weights }
            output_b = self._model.build_forward_pass(input_b, new_fast_weights)
            loss_b, test_loss = self._model.build_loss(label_b, output_b)    # (5,10,1) 들어가니까 task 당 들어가고 있음 그럼 loss 출력이 (5,)로 나와야함 : test_loss에 담겨있음
            acc_b = self._model.build_accuracy(label_b, output_b)
        return output_b, loss_a, acc_a, loss_b, acc_b, new_fast_weights, test_loss    # 근대 실제로는 저 5가 다 평균내져서 나옴;;

    def _compute_metatrain_and_acc(self, handle):
        loss_a, acc_a, losses_b, accs_b, _ = self._sess.run([self._loss_a, self._acc_a, self._losses_b, self._accs_b, self._metatrain_op], feed_dict={self._handle: handle})
        return loss_a, acc_a, losses_b, accs_b

    def _compute_metatrain(self, handle):
        loss_a, losses_b, _ = self._sess.run([self._loss_a, self._losses_b, self._metatrain_op], feed_dict={self._handle: handle})
        return loss_a, losses_b

    def _compute_metatest_and_acc(self, handle): 
        return self._sess.run([self._input_b, self._outputsb, self._loss_a, self._acc_a, self._losses_b, self._accs_b], feed_dict={self._handle: handle})

    def _compute_metatest(self, handle):
        return self._sess.run([self._input_b, self._outputsb, self._loss_a, self._losses_b, self._test_losses, self._real_a, self._real_b], feed_dict={self._handle: handle})     
            
    def test(self, test_itr, restore_model_path):
        assert restore_model_path

        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())        
        self._model.restore_model(restore_model_path)
        
        _, test_handle = self._metasampler.init_iterators(self._sess)
        self._test(test_itr, test_handle, 50)    # global_step을 50의 배수로 놔야 test_loss 출력됨 -- itr 그래서 50으로 나옴

    def train(self, train_itr, test_itr, test_interval, restore_model_path=None):   # log_image는 뭐지...? 생성된 이미지 저장
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)

        train_handle, test_handle = self._metasampler.init_iterators(self._sess)

        for i in range(train_itr):   # train_itr가 2000
            try:
                if self._compute_acc:
                    loss_a, acc_a, losses_b, accs_b = self._compute_metatrain_and_acc(train_handle)                    
                    acc_a = np.mean(acc_a)
                    accs_b = np.array(accs_b).mean(axis=1)                    
                else:
                    loss_a, losses_b = self._compute_metatrain(train_handle)                    

                loss_a = np.mean(loss_a)
                # print(np.array(losses_b).shape)   # (6,7)이 나옴
                losses_b = np.array(losses_b).mean(axis=1)   # 각 task 즉, class 당 평균냄
                # print(losses_b.shape)             # (6,)
                self._logger.new_summary()
                self._logger.add_value("train_loss_a", loss_a)
                self._logger.add_value("train_loss_b/update_", losses_b.tolist())
                if self._compute_acc:
                    self._logger.add_value("train_acc_a", acc_a)
                    self._logger.add_value("train_acc_b/update_", accs_b.tolist())
                self._logger.dump_summary(i)
                self._logger.save_tf_variables(self._model.get_variables(), i, self._sess)
            except tf.errors.OutOfRangeError:
                self._metasampler.restart_train_dataset(self._sess)

            #if i % test_interval == 0:
            #    self._test(test_itr, test_handle, i)        # train에서도 100번 마다 test를 시행함 -- 나는 다 끝나고 한번에 test 성능 보고싶어  

        self._logger.close()

    def _test(self, test_itr, test_handle, global_step):
        self._metasampler.restart_test_dataset(self._sess)
        total_loss_a = 0
        total_losses_b = np.array([0.] * self._num_updates)      # update 횟수만큼 b의 loss 출력              
        for j in range(test_itr):
            try:
                if self._compute_acc:   # input imgs는 (7, 5, 10) 원래 꼴
                    input_imgs, gen_imgs, loss_a, acc_a, losses_b, accs_b = self._compute_metatest_and_acc(test_handle)                    
                    acc_a = np.mean(acc_a)
                    accs_b = np.array(accs_b).mean(axis=1)                    
                else:
                    input_imgs, gen_imgs, loss_a, losses_b, test_losses, real_a, real_b = self._compute_metatest(test_handle) # 최종으로 받을때는 test_losses로 이름 바꿔줌
                
                print("********************")
                print(real_b)
                #print(np.array(test_losses).shape)         # 이게 (9, 7, 5)니까 맨 마지막 losses만 가져오면 됨~
                this_loss = np.array(test_losses)[-1]       # 나는 여기서 내가 원하는 것만 떼온다 (7, 5)꼴
                loss_a = np.mean(loss_a)                    # 나는 num_updates의 맨 마지막 최종 업데이트본 가지고 예측
                losses_b = np.array(losses_b).mean(axis=1)  # mean(axis=1)이니까 7개를 다 더해서 평균낸다.
                # 난 이렇게 평균낸 (20,)가 필요한게 아니라 (1,7)과 같이 맨 끝의 모든 7개 각각의 loss가 필요하기 때문에 위위위와 같이 처리
                total_loss_a += loss_a
                total_losses_b += losses_b
            except tf.errors.OutOfRangeError:
                self._metasampler.restart_test_dataset(self._sess)
  
        #total_loss_a = total_loss_a / test_itr
        #total_losses_b = total_losses_b / test_itr
        self._logger.new_summary()                 
        self._logger.add_value("test_loss_a", total_loss_a / test_itr)
        self._logger.add_value("test_loss_b/update_", (total_losses_b / test_itr).tolist())
        if self._compute_acc:
            self._logger.add_value("test_acc_a", acc_a)
            self._logger.add_value("test_acc_b/update_", accs_b.tolist())
        print(input_imgs.shape)
        
        test_loss = np.percentile(self.all_losses, (1 - self.outliers_fraction)*100)  # all_losses가 train에서 저장된 거를 그냥 수동으로 저장
        print("Reconstruction threshold : ", test_loss)
        print(this_loss)
        y_pred = []
        for i in range(input_imgs.shape[0]):
            return_label = []
            for j in range(input_imgs.shape[1]):
                if this_loss[i][j] < test_loss:
                    return_label.append(0)        # 정상
                else:
                    return_label.append(1)        # 비정상
            y_pred.append(return_label)
            pd.DataFrame(return_label).to_csv(str(i)+'_결과.csv')

        y_true = real_b.reshape(-1,)
        y_pred = np.array(y_pred).reshape(-1,)
        cmap = plt.cm.binary
        labels= ['normal','anomaly']
        cm = confusion_matrix(y_true, y_pred)
        tick_marks = np.array(range(len(labels))) + 0.5
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(4, 2), dpi=120)
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        intFlag = 0 
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            if (intFlag):
                c = cm[y_val][x_val]
                plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')
            else:
                c = cm_normalized[y_val][x_val]
                if (c > 0.01):
                    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
                else:
                    plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
        if(intFlag):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
        else:
            plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Meta_VAE Confusion-Matrix')
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels)
        plt.yticks(xlocations, labels)
        plt.ylabel('Index of True Classes')
        plt.xlabel('Index of Predict Classes')
        plt.show()
        self._logger.dump_summary(global_step)
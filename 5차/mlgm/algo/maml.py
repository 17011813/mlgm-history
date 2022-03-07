import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
from csv import writer
from datetime import datetime

class Maml:
    def __init__(self,
                 model,         # 여기서 model은 VAE 받아온다
                 metasampler,
                 sess,
                 logger,
                 num_updates=1,
                 update_lr=0.0001,
                 meta_lr=0.0001,
                 outliers_fraction=0.07):           # loss_b의 num_updates를 한번만 하기 위해 여기서 세팅해줌
        self._model = model
        self._metasampler = metasampler
        self._sess = sess
        self._num_updates = num_updates        
        self._update_lr = update_lr
        self._meta_lr = meta_lr
        self._logger = logger 
        self._init_variables()
        self._build()        
        self._logger.add_graph(self._sess.graph)
        self.outliers_fraction = outliers_fraction

    def _build(self):        
        with self._sess.graph.as_default():   # (3, 5, 8) (3, 5, 8, 1) (3, 5, 8) (3, 5, 8, 1) (3, 5) (3, 5)
            self._input_a, self._label_a, self._input_b, self._label_b, self._real_a, self._real_b = self._metasampler.build_inputs_and_labels(self._handle)
            #print(self._input_a.shape, self._label_a.shape, self._input_b.shape, self._label_b.shape, self._real_a.shape, self._real_b.shape)
            def task_metalearn(args):
                input_a, label_a, input_b, label_b = args
                loss_a, mse_loss, print_test_loss , f_w = None, None, None, None
                losses_b, outputs_b = [], []

                # build_update에서 받아온 query set(b)의 loss_b가 그냥 바로 출력되어야함. -- 근데 1번만 도는것도 update인가..? 그냥 돌아서 loss 계산하는거랑 gradietn update는 다른건가?
                for i in range(self._num_updates):       # print(label_b.shape)  # (5,10,1)
                    # f_w 즉, fast weight를 가지고 
                    output_b, loss, loss_b, f_w, mse_loss, test_loss = self._build_update(input_a, label_a, input_b, label_b, self._update_lr, f_w)
                    if loss_a is None:                        
                        loss_a = tf.math.reduce_mean(loss) 
                    if mse_loss is None:                        
                        mse_loss = tf.math.reduce_mean(mse_loss)         # loss_a랑 같이 하나의 값이기 때문에 losses_b처럼 처리할 이유가 없다.  
                    
                    if i==0:   # 맨 처음꺼만 가져다가 test_loss로 사용
                        print_test_loss = test_loss
                        
                    outputs_b.append(output_b)
                    losses_b.append(tf.math.reduce_mean(loss_b))   # 평균냄 -->  ()로 나옴

                    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")        

                return outputs_b, loss_a, losses_b, mse_loss, print_test_loss

            out_dtype = ([tf.float32] * self._num_updates, tf.float32, [tf.float32] * self._num_updates, tf.float32, tf.float32)  # tf.float32 차원을 붙여서 self._mse_loss를 받자
            elems = (self._input_a, self._label_a, self._input_b, self._label_b)
            self._outputsb, self._loss_a, self._losses_b, self._mse_loss, self._test_loss = tf.map_fn(
                        task_metalearn,
                        elems=elems,
                        dtype=out_dtype,
                        parallel_iterations=self._metasampler.meta_batch_size)

            with tf.variable_scope("metatrain", values=[self._losses_b]):                   # outer meta parameter는
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
        loss_a, loss_b = None, None

        with tf.variable_scope("update", values=values):    # update할 때 a로 하는거 보니까 a가 meta parameter
            output_a = self._model.build_forward_pass(input_a, fast_weights)
            loss_a, mse_loss, _ = self._model.build_loss(label_a, output_a)        # a가 support set이니까 여기서 mse threshold 가져옴
            grads, weights = self._model.build_gradients(loss_a, fast_weights)     # a만 gradient build 하는거 보니까 meta 맞네__아님
            with tf.variable_scope("fast_weights", values=[weights, grads]):
                new_fast_weights = { w: weights[w] - update_lr * grads[w] for w in weights }  # loss_a에서 나온 weight를 가지고
            output_b = self._model.build_forward_pass(input_b, new_fast_weights)              # query set인 input_b에서 출력 뽑음
            loss_b, _, test_loss = self._model.build_loss(label_b, output_b)       # (5,10,1) 들어가니까 task 당 들어가고 있음 그럼 loss 출력이 (5,)로 나와야함 : test_loss에 담겨있음
        return output_b, loss_a, loss_b, new_fast_weights, mse_loss, test_loss     # 근데 실제로는 저 5가 다 평균내져서 나옴;;

    def _compute_metatrain(self, handle):
        loss_a, losses_b, _ = self._sess.run([self._loss_a, self._losses_b, self._metatrain_op], feed_dict={self._handle: handle})
        return loss_a, losses_b   # metatrain에서는 필요한 애만 return

    def _compute_metatest(self, handle):
        return self._sess.run([self._input_b, self._outputsb, self._loss_a, self._losses_b, self._mse_loss, self._test_loss, self._real_a, self._real_b], feed_dict={self._handle: handle})     
            
    def test(self, test_itr, restore_model_path):
        assert restore_model_path
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())        
        self._model.restore_model(restore_model_path)
        
        _, test_handle = self._metasampler.init_iterators(self._sess)
        self._test(test_itr, test_handle, 50)    # global_step을 50의 배수로 놔야 test_loss 출력됨 -- itr 그래서 무조건 50으로 나옴

    def train(self, train_itr, test_itr, test_interval, restore_model_path=None):   # log_image는 뭐지...? 생성된 이미지 저장
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if restore_model_path:
            self._model.restore_model(restore_model_path)

        train_handle, test_handle = self._metasampler.init_iterators(self._sess)

        for i in range(train_itr):   # 20,000번 돌면서
            try:                                  # train_handle은 meta batch_N (3)당 input_K (5) * 2 해서 (30, 8)과 (30,)가 들어가서 계산되어 나옴
                loss_a, losses_b = self._compute_metatrain(train_handle)   # train_handle은 하나의 task         
                #print(loss_a.shape)       # (3, ) 한 task의 support set내에서 N(3)개 각각 class의 loss가 나옴
                loss_a = np.mean(loss_a)
                #print(loss_a.shape)       # ()    이걸 한 task의 support set당 loss로 평균내서 최종 loss_a
                #print(np.array(losses_b).shape)           # 각 업데이트내의 각 task들의 loss인 (num_updates, 3)이 나옴
                losses_b = np.array(losses_b).mean(axis=1) # 각 task 즉, class들을 평균내서 각 업데이트 당 하나의 최종 loss를 출력 -- (num_updates,)
                self._logger.new_summary()
                self._logger.add_value("train_loss_a", loss_a)
                self._logger.add_value("train_loss_b/update_", losses_b.tolist())   # 한 train iter내에서 왜 train set이 하나씩? 362개가 아니라
                self._logger.dump_summary(i, self._model.get_variables(), self._sess)
            except tf.errors.OutOfRangeError:
                self._metasampler.restart_train_dataset(self._sess)

            #if i % test_interval == 0:
            #    self._test(test_itr, test_handle, i)        # train에서도 100번 마다 test를 시행함 -- 나는 다 끝나고 한번에 test 성능 보고싶어  

        self._logger.close()

    def _test(self, test_itr, test_handle, global_step):
        self._metasampler.restart_test_dataset(self._sess)
        #total_loss_a = 0
        #total_losses_b = np.array([0.] * self._num_updates)      # update 횟수만큼 b의 loss 출력  -- 사실 test일때는 1개 인거지
        y_true, y_pred = [], []            
        for j in range(test_itr):      # input imgs는 (3, 5, 8) 원래 꼴
            try:        # (5, 3, 5, 8, 1) gen_imgs는 input_b (query set)으로 부터 얻은 출력값
                input_imgs, gen_imgs, loss_a, losses_b, mse_loss, test_loss, real_a, real_b = self._compute_metatest(test_handle) # 에초에 test_handle이 batch로 들어감
                #print("**************************************real b : ",real_b)

                loss_a = np.mean(loss_a)                     # (3,) --> ()
                #mse_loss = np.mean(mse_loss)                # (3, 100) --> ()  이렇게 굳이 안해도 밑에 threshold될때 알아서 됨
                #print(np.array(test_loss).shape)            # (3, 5)   모든 애들 loss 각각 출력
                #print(losses_b.shape)                       # (5, )

                # 여기서부터는 원래 for 문 바깥에 있었음
                self._logger.new_summary()                 
                self._logger.add_value("test_loss_a", loss_a)
                self._logger.add_value("test_loss_b/update_", np.array(losses_b).mean(axis=1).tolist())    # test_itr없이 매 번 출력인거지
                #print(total_loss_a.shape, total_losses_b.shape)
 
                threshold = np.percentile(mse_loss, (1 - self.outliers_fraction)*100)  # loss_a에서 mse loss가 Meta test의 support set의 loss기 때문에 얘가 맞음
                print("\nReconstruction threshold : {:.4f}".format(threshold))
                print("\n",test_loss,"\n")    # 원래 this loss 하나씩 가져오던거
                tmp_y_pred = []                           # input_imgs.shape  (3, 5, 8)
                for i in range(input_imgs.shape[0]):      # 3
                    return_label = []
                    for j in range(input_imgs.shape[1]):  # 5
                        if test_loss[i][j] < threshold:
                            return_label.append(0)        # 정상
                        else:
                            return_label.append(1)        # 비정상
                    tmp_y_pred.append(return_label)
                    pd.DataFrame(return_label).to_csv(str(i)+'번째결과.csv')

                y_true.append(real_b.reshape(-1,))
                y_pred.append(np.array(tmp_y_pred).reshape(-1,))  # 둘 다 (3, 15)
                self._logger.dump_summary(global_step, self._model.get_variables(), self._sess)    # iter마다 출력하도록 바꿈
            except tf.errors.OutOfRangeError:
                self._metasampler.restart_test_dataset(self._sess)

        y_true, y_pred = np.array(y_true).reshape(-1,), np.array(y_pred).reshape(-1,)

        cmap = plt.cm.binary
        labels= ['normal','anomaly']
        print(classification_report(y_true, y_pred, target_names=labels))

        save = {f1_score(y_true, y_pred), recall_score(y_true, y_pred), precision_score(y_true, y_pred), accuracy_score(y_true, y_pred),threshold, datetime.now().strftime("%m/%d_%H:%M")}
        with open('./정확도.csv', 'a') as f_object:  
            writer_object = writer(f_object)
            writer_object.writerow(save)
            f_object.close()

        tick_marks = np.array(range(len(labels))) + 0.5
        np.set_printoptions(precision=2)
        cm_normalized = confusion_matrix(y_true, y_pred).astype('float') / confusion_matrix(y_true, y_pred).sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(4, 2), dpi=120)
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Meta_VAE Confusion-Matrix')
        plt.colorbar()
        plt.xticks(np.array(range(len(labels))), labels)
        plt.yticks(np.array(range(len(labels))), labels)
        plt.ylabel('Index of True Classes')
        plt.xlabel('Index of Predict Classes')
        plt.show()
        

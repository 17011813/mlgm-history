from datetime import datetime
import tensorflow as tf
from tensorflow import summary as summ

class Logger:
    def __init__(self, exp_name, graph=None, std_out_period=100):
        self._log_path = ("data/" + exp_name + "/" + exp_name + "_" + datetime.now().strftime("%m_%d_%H_%M"))  # 월 일 시 분
        self._writer = summ.FileWriter(self._log_path, graph)
        self._summary_mrg = None
        self._writer.flush()
        self._std_out_period = std_out_period
        self.min = 5       # 보다 작아야 ckpt 저장하도록

    def new_summary(self):
        self._summary = tf.Summary()
        self._std_out = {}

    def add_value(self, name, value):
        if isinstance(value, list):    # value가 list 형이면 True
            for i, val in enumerate(value):
                name_id = name + "{}".format(i)
                self._summary.value.add(tag=name_id, simple_value=val)
                self._std_out.update({name_id: val})
        else:
            self._summary.value.add(tag=name, simple_value=value)
            self._std_out.update({name: value})

    def add_graph(self, graph):
        self._writer.add_graph(graph)
    
    def add_image(self, image, itr):
        self._writer.add_summary(image, itr)

    @property
    def summary(self):
        return self._summary_mrg

    def dump_summary(self, itr, var_list, sess):
        self._writer.add_summary(self._summary, itr)
        self._writer.flush()
        if not (itr % self._std_out_period) and itr > 0:     # _std_out_period가 50이니까 50마다 출력함
            print("------------------------------------------")
            print("exp_name: {}".format(self._log_path))
            print("itr: {}".format(itr))
            for k, v in self._std_out.items():
                print("{}: {:.4f}".format(k, v))   # loss 소수점 떼버리기~가 아니라 출력 다해

                if v < self.min:     # update loss가 제일 작을 때 ckpt 저장
                    self.min = v 
                    saver = tf.train.Saver(var_list)
                    saver.save(sess, self._log_path + "/itr_{}_loss{:.4f}".format(itr,self.min))

    def close(self):
        self._writer.close()
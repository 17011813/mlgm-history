from itertools import permutations
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from mlgm.sampler import MetaSampler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
class MnistMetaSampler(MetaSampler):
    def __init__(
            self,
            batch_size,
            meta_batch_size,
            train_digits,
            test_digits
    ):
        assert train_digits is None or (type(train_digits) == list and [2 <= digit <= 8 for digit in train_digits])
        assert test_digits is None or (type(test_digits) == list and [0 <= digit <= 1 for digit in test_digits])     # test는 0과 1로만 구성
        self._train_digits, self._test_digits = list(set(train_digits)), list(set(test_digits))

        meta_train, meta_test = pd.read_csv('./data/train.csv'), pd.read_csv('./data/test.csv')

        self.train_inputs, train_labels = scaler.fit_transform(np.array(meta_train.iloc[:,:-1])), np.array(meta_train.iloc[:,-1]).reshape(-1,)
        self.test_inputs, test_labels = scaler.transform(np.array(meta_test.iloc[:,:-1])), np.array(meta_test.iloc[:,-1]).reshape(-1,)
           
        self._train_inputs_per_label, self._test_inputs_per_label = {}, {}    # 라벨당 train input의 id
        self._train_size, self._test_size = 0, 0                        # train과 test 크기는 처음에 0으로 시작해서

        for digit in self._train_digits:                        # 2부터 8까지 하나씩 돌면서
            ids = np.where(digit == train_labels)[0]            # train label 내에서만 해당 label 위치를 id로 찾자
            self._train_size += len(ids)                        # 해당 class(digit)랑 같은 데이터 길이만큼 train이 됨..
            random.shuffle(ids)
            self._train_inputs_per_label.update({digit: ids})      #  {0:25,36,57 ...} 이런식으로 각 task 숫자의 index ids위치

        for digit in self._test_digits:                         # test dataset 내에서 0일 때랑 1일 때랑 for문 두번 돈다
            ids = np.where(digit == test_labels)[0]             # 그래서 test dataset 내에서 0일때 ids 저장해주고, 1일때 ids 저장
            self._test_size += len(ids)                         
            random.shuffle(ids)
            self._test_inputs_per_label.update({digit: ids})       # 여기서 digit당 ids를 저장해주는게 문제 ㄴㄴ 중복만 안되면 ㄱㅊ

        super().__init__(batch_size, meta_batch_size)

    def _gen_dataset(self, test=False):     # 여기서는 (362, 30, 8) (362, 30) // test (39, 30, 8) (39, 30) 잘 나가는데
        digits = self._test_digits if test else self._train_digits
        inputs_per_label = self._test_inputs_per_label if test else self._train_inputs_per_label  # {0: 1, 5, 7..}, {1: 56, 2 ...}
        inputs = self.test_inputs if test else self.train_inputs
        
        tasks = []
        while True:
            tasks_remaining = self._meta_batch_size - len(tasks)
            if tasks_remaining <= 0:
                break            
            tasks_to_add = list(permutations(digits, 1))   # permutation 함수 때문에 0 1 왔다갔다
            n_tasks_to_add = min(len(tasks_to_add), tasks_remaining)    # digits는 2-7까지 숫자
            tasks.extend(tasks_to_add[:n_tasks_to_add])
        
        num_inputs_per_meta_batch = (self._batch_size  * self._meta_batch_size)  # 200 (=100*2) * 6 : 한 iter당 meta_batch(task)는 6개이고 각각 100개씩
        ids, lbls = np.empty((0, num_inputs_per_meta_batch), dtype=np.int32), np.empty((0, num_inputs_per_meta_batch), dtype=np.int32)
        data_size = self._test_size if test else self._train_size         # 트레인 (56430)                 테스트 (4312)
        data_size = min(data_size // num_inputs_per_meta_batch, 1000)     # 트레인 (56430 // 1200) __ 47   테스트 (4312 // 1200) __ 3
        
        for i in range(data_size): # 한번 데이터셋을 총 564개 만큼 돌면서
            all_ids, all_labels = np.array([], dtype=np.int32), np.array([], dtype=np.int32) 
            for task in tasks:   # 6번 도는데 tasks는 (train) [(2,), (3,), (4,), (5,), (6,), (7,)] (test) [(0,), (1,), (0,), (1,), (0,), (1,)]
                task_ids, task_labels = np.array([], dtype=np.int32), np.array([], dtype=np.int32)

                for i, label in enumerate(task):
                    label_ids = np.random.choice(inputs_per_label[label], self._batch_size)
                    labels = np.empty(self._batch_size, dtype=np.int32)
                    labels.fill(i)
                    task_labels = np.append(task_labels, labels)
                    task_ids = np.append(task_ids, label_ids) 
                all_labels, all_ids = np.append(all_labels, task_labels), np.append(all_ids, task_ids)

            ids = np.append(ids, [all_ids], axis=0)        # 인덱스 [44294 21837   726 ... 35697 10132 23818] (6, 40) (1, 40)
            lbls = np.append(lbls, [all_labels], axis=0)   # 실제 정답 [0 0 0 ... 0 0 0] (6, 40) (1, 40)
        all_ids_sym = tf.convert_to_tensor(ids)
        inputs_sym = tf.convert_to_tensor(inputs, dtype=tf.float32)
        all_inputs = tf.gather(inputs_sym, all_ids_sym)                     # 한 iteration당 4 * 5 * 2 해서 40개인데 각각 feature 30개씩
        all_labels = tf.convert_to_tensor(lbls, dtype=tf.dtypes.int32)      # 각 데이터의 0인지 1인지 실제 라벨도 40개 있다.
        #print("에헤라디야에헤라디야에헤라디야에헤라디야",all_inputs.shape, all_labels.shape)
        return tf.data.Dataset.from_tensor_slices((all_inputs, all_labels))    # train (47, 1200, 8) (47, 1200) // test (3, 1200, 8) (3, 1200)

    def build_inputs_and_labels(self, handle):
        slice_size = (self._batch_size // 2) 
        input_batches, label_batches = self._gen_metadata(handle)    # 실제 정답 label 값 여기서 받아와서 acc 출력 가능~ (6, 200, 8) (6, 200)
        
        input_a = tf.slice(input_batches, [0, 0, 0], [-1, slice_size, -1])
        input_b = tf.slice(input_batches, [0, slice_size, 0], [-1, -1, -1])   # 이렇게 가져와서 support_a랑 query_b랑 반으로 나눔~

        # VAE는 input이 label이랑 같아서 항상 True라 same_input_and_label 지움
        label_a = tf.reshape(input_a, input_a.get_shape().concatenate(1))   # get_shape하면 (7,5,10) 차원 맞추려고 끝에 1 붙임
        label_b = tf.reshape(input_b, input_b.get_shape().concatenate(1))   # 그래서 보면 label에 input이 그대로 들어감

        real_a = tf.slice(label_batches, [0, 0], [-1, slice_size])
        real_b = tf.slice(label_batches, [0, slice_size], [-1, -1])
        #print("에헤라디야에헤라디야에헤라디야에헤라디야",input_a.shape, label_a.shape, input_b.shape, label_b.shape, real_a.shape, real_b.shape)
        return input_a, label_a, input_b, label_b, real_a, real_b  # (6, 100, 8) (6, 100, 8, 1) (6, 100, 8) (6, 100, 8, 1) (6, 100) (6, 100)
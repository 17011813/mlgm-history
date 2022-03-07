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
            train_digits,   # 0의 list로 들어옴
            test_digits
    ):
        assert train_digits is None or (type(train_digits) == list and [0 <= digit <= 0 for digit in train_digits])   # train은 0으로만 구성
        assert test_digits is None or (type(test_digits) == list and [0 <= digit <= 1 for digit in test_digits])     # test는 0과 1로만 구성
        
        self._train_digits = list(set(train_digits))
        self._test_digits = list(set(test_digits))

        train_inputs, train_labels = np.array(pd.read_csv('./cover_train.csv')), np.array(pd.read_csv('./cover_train_label.csv'))
        test_inputs, test_labels = np.array(pd.read_csv('./cover_test.csv')), np.array(pd.read_csv('./cover_test_label.csv'))
        train_labels, test_labels = train_labels.reshape(-1,), test_labels.reshape(-1,)  # MNIST랑 형태 맞추기 위해 마지막에 1 추가

        train_inputs = scaler.fit_transform(train_inputs)   # 정규화 하나로 이렇게 달라지다니
        test_inputs = scaler.transform(test_inputs)
# MNIST는 train과 test가 따로 들어오는 구조라 일단 합쳐준 후에, 섞고서 0부터 9까지 각 class 별로 또 나누는 작업 들어감 -- 내꺼로 그래서 수정해야함
        inputs = np.concatenate((train_inputs, test_inputs))   # Anomaly인데 합쳐도 되낭.... 섞는건 아니겠지...?  ㅇㅇ 섞음!!!
        labels = np.concatenate((train_labels, test_labels))   # (227300, 10) (227300,) (58746, 10) (58746,)로 들어감

        self._train_inputs_per_label, self._test_inputs_per_label = {}, {}    # 라벨이 왜 필요하나면 라벨이 input과 같음 True
        self._train_size, self._test_size = 0, 0   # train과 test 크기는 처음에 0으로 시작해서

        for digit in self._train_digits:
            ids = np.where(digit == labels)[0]  
            self._train_size += len(ids)     # 해당 class(digit)랑 같은 데이터 길이만큼 train이 됨...
            #random.shuffle(ids)             # 즉, 원래 MNIST는 0-6 라벨은 다 train으로 들어가고, 7-9 라벨은 다 test data로 들어감
            self._train_inputs_per_label.update({digit: ids})   # 라벨당 train input인데 나는 다 0이라...

        for digit in self._test_digits:
            ids = np.where(digit == labels)[0]
            self._test_size += len(ids)
            #random.shuffle(ids)
            self._test_inputs_per_label.update({digit: ids})
        print("트레인 테스트 사이즈: ",self._train_size, self._test_size)

        super().__init__(batch_size, meta_batch_size, inputs)

    def _gen_dataset(self, test=False):
        digits = self._test_digits if test else self._train_digits
        inputs_per_label = self._test_inputs_per_label if test else self._train_inputs_per_label

        tasks = []
        while True:
            tasks_remaining = self._meta_batch_size - len(tasks)
            if tasks_remaining <= 0:
                break            
            tasks_to_add = list(permutations(digits, 1))
            n_tasks_to_add = min(len(tasks_to_add), tasks_remaining)
            tasks.extend(tasks_to_add[:n_tasks_to_add])            

        num_inputs_per_meta_batch = (self._batch_size  * self._meta_batch_size)
        
        ids = np.empty((0, num_inputs_per_meta_batch), dtype=np.int32)
        lbls = np.empty((0, num_inputs_per_meta_batch), dtype=np.int32)

        data_size = self._test_size if test else self._train_size
        data_size = data_size // num_inputs_per_meta_batch
        data_size = min(data_size, 1000)
                        
        for i in range(data_size):
            all_ids = np.array([], dtype=np.int32)
            all_labels = np.array([], dtype=np.int32)
            for task in tasks:
                task_ids = np.array([], dtype=np.int32)
                task_labels = np.array([], dtype=np.int32)
                for i, label in enumerate(task):
                    label_ids = np.random.choice(inputs_per_label[label], self._batch_size)
                    labels = np.empty(self._batch_size, dtype=np.int32)
                    labels.fill(i)
                    task_labels = np.append(task_labels, labels)
                    task_ids = np.append(task_ids, label_ids)
                all_labels = np.append(all_labels, task_labels)
                all_ids = np.append(all_ids, task_ids)
            ids = np.append(ids, [all_ids], axis=0)
            lbls = np.append(lbls, [all_labels], axis=0)

        all_ids_sym = tf.convert_to_tensor(ids)
        inputs_sym = tf.convert_to_tensor(self._inputs, dtype=tf.float32)
        all_inputs = tf.gather(inputs_sym, all_ids_sym)
        all_labels = tf.convert_to_tensor(lbls, dtype=tf.dtypes.int32)
        dataset_sym = tf.data.Dataset.from_tensor_slices((all_inputs, all_labels))
        print("dataset_sym", dataset_sym)
        return dataset_sym   # (70, 10) (10,)

    def build_inputs_and_labels(self, handle):
        slice_size = (self._batch_size // 2) 
        input_batches, _ = self._gen_metadata(handle)
        
        input_a = tf.slice(input_batches, [0, 0, 0], [-1, slice_size, -1])
        input_b = tf.slice(input_batches, [0, slice_size, 0], [-1, -1, -1])

        # VAE는 input이 label이랑 같아서 항상 True라 same_input_and_label 지움
        label_a = tf.reshape(input_a, input_a.get_shape().concatenate(1))   # get_shape하면 (7,5,10) 차원 맞추려고 끝에 1 붙임
        label_b = tf.reshape(input_b, input_b.get_shape().concatenate(1))   # 그래서 보면 label에 input이 그대로 들어감

        #print("----------------------",input_a.shape, label_a.shape, input_b.shape, label_b.shape)
        return input_a, label_a, input_b, label_b
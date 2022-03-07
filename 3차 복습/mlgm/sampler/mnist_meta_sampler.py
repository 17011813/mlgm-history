from itertools import permutations
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
        assert train_digits is None or (type(train_digits) == list and [0 <= digit <= 0 for digit in train_digits])   # train은 0으로만 구성
        assert test_digits is None or (type(test_digits) == list and [0 <= digit <= 1 for digit in test_digits])     # test는 0과 1로만 구성
        self._train_digits, self._test_digits = list(set(train_digits)), list(set(test_digits))

        self.train_inputs, train_labels = scaler.fit_transform(np.array(pd.read_csv('./data/0_train.csv'))), np.array(pd.read_csv('./data/0_train_label.csv')).reshape(-1,)
        self.test_inputs, test_labels = scaler.transform(np.array(pd.read_csv('./data/20_test.csv'))), np.array(pd.read_csv('./data/20_test_label.csv')).reshape(-1,)
           
        self._train_ids_per_label, self._test_ids_per_label = {}, {}    # 라벨당 train input인데 나는 다 0이라
        self._train_size, self._test_size = 0, 0   # train과 test 크기는 처음에 0으로 시작해서

        for digit in self._train_digits:                        # 0 하나밖에 없어서 for문 한번만 돈다
            ids = np.where(digit == train_labels)[0]            # train label 내에서만 0 찾자 -- 다 0임
            self._train_size += len(ids)                        # 해당 class(digit)랑 같은 데이터 길이만큼 train이 됨..
            self._train_ids_per_label.update({digit: ids})      #  {0:25,36,57 ...} 이런식으로 각 task 숫자의 index ids위치 -- 순서대로 저장

        for digit in self._test_digits:                         # test dataset 내에서 0일 때랑 1일 때랑 for문 두번 돈다
            ids = np.where(digit == test_labels)[0]             # 그래서 test dataset 내에서 0일때 ids 저장해주고, 1일때 ids 저장
            self._test_size += len(ids)                         # 원래 random으로 ids 섞어주는데 나는 안했다... 다시 해야하나?
            self._test_ids_per_label.update({digit: ids})       # 여기서 digit당 ids를 저장해주는게 문제임 걍 섞어야하는데

        super().__init__(batch_size, meta_batch_size)

    def _gen_dataset(self, test=False):
        digits = self._test_digits if test else self._train_digits
        ids_per_label = self._test_ids_per_label if test else self._train_ids_per_label  # {0: 1, 5, 7..}, {1: 56, 2 ...}
        inputs = self.test_inputs if test else self.train_inputs
        
        tasks = []
        while True:
            tasks_remaining = self._meta_batch_size - len(tasks)
            if tasks_remaining <= 0:
                break            
            tasks_to_add = list(permutations(digits, 1))   # permutation 함수 때문에 0 1 왔다갔다
            n_tasks_to_add = min(len(tasks_to_add), tasks_remaining)
            tasks.extend(tasks_to_add[:n_tasks_to_add])

        num_inputs_per_meta_batch = (self._batch_size  * self._meta_batch_size)  # (100*2) * 3
        ids, lbls = np.empty((0, num_inputs_per_meta_batch), dtype=np.int32), np.empty((0, num_inputs_per_meta_batch), dtype=np.int32)
        data_size = self._test_size if test else self._train_size         # 트레인 (23875)              테스트 (1090)
        # print("어링하ㅓㄴ마러암너라미ㅓㅇㄹ니마ㅓㅏㅣㄹㅇㅁㄴ",data_size)  # 트레인 23875랑 테스트 1090 아주 잘 맞습니다 ^^
        data_size = min(data_size // num_inputs_per_meta_batch, 1000)     # 트레인 (23875 // 300) __ 39 테스트 (1090 // 300) __ 1 아주 맞음
        #print("어링하ㅓㄴ마러암너라미ㅓㅇㄹ니마ㅓㅏㅣㄹㅇㅁㄴ",data_size)  
        for i in range(data_size): # 한번 데이터셋을 총 1000개 만큼 돌면서 -- 왜 1000 만큼 도는거지? 하나만해도 훈련 가능하잖아; 왜지?
            all_ids, all_labels = np.array([], dtype=np.int32), np.array([], dtype=np.int32) 
            for task in tasks:   # 4번 도는데 tasks는 [(0,), (0,), (0,), (0,)] 6개 (train set) [(0,), (1,), (0,), (1,)] 1개 (test set)
                task_ids, task_labels = np.array([], dtype=np.int32), np.array([], dtype=np.int32)
                labels = np.empty(self._batch_size, dtype=np.int32)
                label_ids = np.random.choice(ids_per_label[task[0]], self._batch_size)   # 여기서 0이랑 1 순서 섞이는게 아니라 0 안에서 랜덤 추출
                labels.fill(task[0])    # 원래 i였는데 i가 아니라 label(task[0])로 넣어야하지 않나? 내가 바꿔줌 -- i로 하면 무조건 0임..
                
                task_labels, task_ids = np.append(task_labels, labels), np.append(task_ids, label_ids) 
                all_labels, all_ids = np.append(all_labels, task_labels), np.append(all_ids, task_ids)

            ids = np.append(ids, [all_ids], axis=0)        # 인덱스 [44294 21837   726 ... 35697 10132 23818] (6, 40) (1, 40)
            lbls = np.append(lbls, [all_labels], axis=0)   # 실제 정답 [0 0 0 ... 0 0 0] (6, 40) (1, 40)
        all_ids_sym = tf.convert_to_tensor(ids)
        inputs_sym = tf.convert_to_tensor(inputs, dtype=tf.float32)
        all_inputs = tf.gather(inputs_sym, all_ids_sym)                     # 한 iteration당 4 * 5 * 2 해서 40개인데 각각 feature 30개씩
        all_labels = tf.convert_to_tensor(lbls, dtype=tf.dtypes.int32)      # 각 데이터의 0인지 1인지 실제 라벨도 40개 있다.
        #print(all_inputs.shape, all_labels.shape)
        return tf.data.Dataset.from_tensor_slices((all_inputs, all_labels))    # train (39, 600, 8) (39, 600) // test (1, 600, 8) (1, 600) 잘 맞구연~

    def build_inputs_and_labels(self, handle):
        slice_size = (self._batch_size // 2) 
        input_batches, label_batches = self._gen_metadata(handle)    # 실제 정답 label 값 여기서 받아와서 acc 출력 가능~ (3, 10, 8) (3, 10)
        
        input_a = tf.slice(input_batches, [0, 0, 0], [-1, slice_size, -1])
        input_b = tf.slice(input_batches, [0, slice_size, 0], [-1, -1, -1])   # 이렇게 가져와서 support_a랑 query_b랑 반으로 나눔~

        # VAE는 input이 label이랑 같아서 항상 True라 same_input_and_label 지움
        label_a = tf.reshape(input_a, input_a.get_shape().concatenate(1))   # get_shape하면 (7,5,10) 차원 맞추려고 끝에 1 붙임
        label_b = tf.reshape(input_b, input_b.get_shape().concatenate(1))   # 그래서 보면 label에 input이 그대로 들어감

        real_a = tf.slice(label_batches, [0, 0], [-1, slice_size])
        real_b = tf.slice(label_batches, [0, slice_size], [-1, -1])
        #print(input_a.shape, label_a.shape, input_b.shape, label_b.shape, real_a.shape, real_b.shape)
        return input_a, label_a, input_b, label_b, real_a, real_b  # (3, 100, 8) (3, 100, 8, 1) (3, 100, 8) (3, 100, 8, 1) (3, 100) (3, 100) 진짜 완벽해
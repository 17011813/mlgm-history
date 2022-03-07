import tensorflow as tf
from tensorflow.keras import layers
from mlgm.algo import Maml
from mlgm.sampler import MnistMetaSampler
from mlgm.model import Vae
from mlgm.logger import Logger

def main():
    metasampler = MnistMetaSampler(
        batch_size=100,   # 이게 K -- 이걸 train 할때는 키우면 안되고, test 할때는 20에서 100정도로 키워도 되나? 그래야 의미있는 성능 평가 아닌가..? 
        meta_batch_size=3,   # 이게 N -- 어차피 1 class니까 1로 하면 안되나? 여기서 class는 밸브다. 그 안은 다 0일 지라도. 지금 데이터 연결해놓음.
        train_digits=list(range(7,10)),      # 정상인 0으로만 구성 -- 근데 지금은 train label 3개!!~
        test_digits=list(range(2))           # label에서 class 0과 1로만 구성 test
        )    
    with tf.Session() as sess:
        model = Vae(
            encoder_layers=[
                layers.Dense(128, activation="relu"),
                layers.Dense(128, activation="relu"),    # 그리고 점점 좁아지는 형태로 가야하는게 아닌가...? 흠
                layers.Dense(units=(16 * 2))    # 32 -- latent dim = 16의 2배로 해줘야 mean 과 stddev 각각 가능
        ],
        decoder_layers=[
                layers.Dense(128, activation="relu"),   # 32 -> 8
                layers.Dense(128, activation="relu"),
                layers.Dense(2, activation="relu"),
                layers.Reshape(target_shape=(2,1)),             # label은 ,1로 한 차원 더 붙어있어서 하나 늘려줘야함
            ], sess=sess)

        logger = Logger("maml_vae")   # 5천마다 저장
        maml = Maml(
            model,
            metasampler,
            sess,
            logger,
            num_updates=5,            # inner gradient 업데이트 횟수 -- 과적합땜에 줄여 5 이상은 의미 없다고 논문
            update_lr=0.0005,
            meta_lr=0.0001,
            outliers_fraction=0.07)   # threshold 설정 비율
        
        maml.train( train_itr=50000, test_itr=1, test_interval=100 )  # iter는 outer 총 예시 task 작업 수 -- 논문에서는 최소 1만 이상,,,?!
        """              
        maml.test( test_itr=10, restore_model_path='./data/maml_vae/maml_vae_01_27_15_28/itr_41450_loss0.1933')               
        """
if __name__ == "__main__":
    main()
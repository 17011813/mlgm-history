import tensorflow as tf
from tensorflow.keras import layers
from mlgm.algo import Maml
from mlgm.sampler import MnistMetaSampler
from mlgm.model import Vae
from mlgm.logger import Logger

def main():
    metasampler = MnistMetaSampler(
        batch_size=5,
        meta_batch_size=7,
        train_digits=list(range(1)),      # 정상인 0으로만 구성 -- label에서 class 0인 애들로만 train 돌림
        test_digits=list(range(2)),   # label에서 class 0과 1로만 구성 test
        )    
    with tf.Session() as sess:
        model = Vae(
            encoder_layers=[
                layers.Dense(128, activation="relu"),
                layers.Dense(128, activation="relu"),    # 그리고 점점 좁아지는 형태로 가야하는게 아닌가...? 흠
                #layers.Dense(units=(20 * 2))    # 32 -- latent dim = 20의 2배로 해줘야 mean 과 stddev 각각 가능
        ],
        decoder_layers=[
                layers.Dense(128, activation="relu"),   # 32 -> 10
                layers.Dense(128, activation="relu"),
                layers.Dense(10, activation="relu"),
                layers.Reshape(target_shape=(10,1)),             # label은 ,1로 한 차원 더 붙어있어서 하나 늘려줘야함
            ],
            sess=sess)

        logger = Logger("maml_vae", save_period=1900)

        maml = Maml(
            model,
            metasampler,
            sess,
            logger,
            num_updates=6,   # 20말고 6정도로 줄여 -- 과적합땜에
            update_lr=0.0005,
            meta_lr=0.0001,
            outliers_fraction=0.07)   # threshold 설정 비율
        
        maml.train( train_itr=1000, test_itr=1, test_interval=100) # 나는 train 중에는 test 안할래
        """              
        maml.test( test_itr=1, restore_model_path='./data/maml_vae/maml_vae_10_19_01_14_22/itr_1900' )               
        """
if __name__ == "__main__":
    main()
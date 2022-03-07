from mlgm.model import Model
import pandas as pd
import numpy as np
import tensorflow as tf

class Vae(Model):
    def __init__(self,
                 sess,
                 encoder_layers,
                 decoder_layers,
                 name="vae"):
        self._encoder = Model(encoder_layers, sess, model_name="encoder")   # encoder랑 decoder instance 선언
        self._decoder = Model(decoder_layers, sess, model_name="decoder")
        self._name = name
        self._sess = sess

    def _encode(self, layer_in, use_tensors=None):   # 여기서 mean이랑 var 받아옴
        mean, logvar = tf.split(self._encoder.build_forward_pass(layer_in, use_tensors), num_or_size_splits=2, axis=1)
        return mean, logvar

    def _reparameterize(self, mean, logvar):      # epsilon 계산 -- 얘도 encoder 포함 -- reparameterize로 z 구한다
        return tf.random.normal(shape=mean.shape) * tf.exp(logvar * .5) + mean

    def _decode(self, z, use_tensors=None):    # 학습된 z를 입력으로 넣고 생성된 예측인 logits를 출력
        logits = self._decoder.build_forward_pass(z, use_tensors)    # z가 _latent_sym임.
        return logits   # 복원 예측된 출력

    @property
    def mean_sym(self):
        return self._mean_sym

    @property
    def logvar_sym(self):
        return self._logvar_sym

    @property
    def latent_sym(self):
        return self._latent_sym

    def build_forward_pass(self, layer_in, use_tensors=None):  # 여기서 나온 output이 예측 값이자 logits가 되는거겠지.
        self._mean_sym, self._logvar_sym = self._encode(layer_in, use_tensors)       # encode에서 평균이랑 분산 구하고
        self._latent_sym = self._reparameterize(self._mean_sym, self._logvar_sym)    # 그걸로 z (_latent_sym) 구한다.
        return self._decode(self._latent_sym, use_tensors)                           # 구한 z를 decode에 넣어서 output얻음

    def build_loss(self, labels, logits):      # 복원된 예측 logits 결과로 라벨과 비교해서 loss 구함    # KL은 (5,)
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self._mean_sym) + tf.square(self._logvar_sym) - tf.log(1e-8 + tf.square(self._logvar_sym)) - 1, 1)
        mse_loss = tf.reduce_sum(tf.square(labels-logits), 1)       # all_loss가 reconstruction error임 -- 얘까지는 (5, 1)인데
        loss = tf.reduce_mean(mse_loss + KL_divergence)             # MLP_VAE 식으로 loss 구하는거 바꿈 (MSE + KLD)  -- 얘부터 () 꼴
        
        mse_loss = tf.reshape(mse_loss, shape = [100,])      # (5,)로 바꿈   -- 2022.01.27에 K를 5에서 100으로 바꿨기 땜에 여기도 바꿔줌
        test_loss = mse_loss + KL_divergence               # (5,)랑 (5,) 끼리 더해서 (5,)꼴
        return loss, mse_loss, test_loss             # mse_loss는 (5,)로 각각 loss 출력함 / test_loss는 (5,)로 모든 애들 loss 출력~  
        # loss가 원래 loss이자 this_loss로 매번 새로운 애, mse_loss가 threshold
    
    def get_variables(self):
        all_vars = []
        all_vars.extend(self._encoder.get_variables())
        all_vars.extend(self._decoder.get_variables())
        return all_vars

    def build_gradients(self, loss_sym, fast_params=None):
        grads = {}
        params = {}
        if not fast_params:
            enc_grads, enc_params = self._encoder.build_gradients(loss_sym, fast_params)
            dec_grads, dec_params = self._decoder.build_gradients(loss_sym, fast_params)
            grads.update(enc_grads)
            grads.update(dec_grads)
            params.update(enc_params)
            params.update(dec_params)
        else:
            grads, params = super(Vae, self).build_gradients(loss_sym, fast_params)
        return grads, params

    def restore_model(self, save_path):
        var_list = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        saver = tf.train.Saver(var_list)
        saver.restore(self._sess, save_path)

import numpy as np
import tensorflow as tf
from random import shuffle
import pathlib
import json
from time import time
from normalizer import Normalizer, Standardizer
import os

class ChemicalTransferNeuralNetwork(object):
    
    def __init__(self, activation='relu', architecture_2b=None, architecture_3b=None, architecture_eam=None,
                 pot_weights=[1., 0.1, 0.001], std_init_weights=0.1, non_negative=None, model_path="model"):
        self.activation = activation
        self.architecture_2b = architecture_2b
        self.architecture_3b = architecture_3b
        self.architecture_eam = architecture_eam
        
        self.std_init_weights = std_init_weights
        self.non_negative = non_negative

        self.pot_weights = pot_weights

        self.fit_first_time = True
        self.is_3b_activated = False

        self.model_path = pathlib.Path(model_path)



    def init_tensors(self):
        
        self.n_weights_2b  = len(self.architecture_2b)  - 1
        self.n_weights_eam = len(self.architecture_eam) - 1
        self.n_weights_3b  = len(self.architecture_3b)  - 1

        architectures = [self.architecture_2b] * 3 + [self.architecture_eam] * 3 + [self.architecture_3b] * 6
        n_features = [a[ 0] for a in architectures]
        n_coefs    = [a[-1] for a in architectures]

        self.Y_ph  =   tf.placeholder(tf.float32, [None], name='Y')
        self.Xs_ph =  [tf.placeholder(tf.float32, [None, n_features[i]], name='X%s' % i) for i in range(12)]
        self.Gs_ph =  [tf.placeholder(tf.float32, [None, n_coefs[i]],  name='G%s' % i) for i in range(12)]

        self.Ws_2b  = [self.init_weight_variable((self.architecture_2b[i], self.architecture_2b[i+1]), name='W_2b%s' % i) 
                      for i in range(self.n_weights_2b)]
        self.Ws_eam = [self.init_weight_variable((self.architecture_eam[i], self.architecture_eam[i+1]), name='W_eam%s' % i) 
                      for i in range(self.n_weights_eam)]
        self.Ws_3b  = [self.init_weight_variable((self.architecture_3b[i], self.architecture_3b[i+1]), name='W_3b%s' % i) 
                      for i in range(self.n_weights_3b)]
        self.Bs_2b  = [self.init_weight_variable((self.architecture_2b[i+1],), name='B_2b%s' % i) 
                      for i in range(self.n_weights_2b)]
        self.Bs_eam = [self.init_weight_variable((self.architecture_eam[i+1],), name='B_eam%s' % i) 
                      for i in range(self.n_weights_eam)]
        self.Bs_3b  = [self.init_weight_variable((self.architecture_3b[i+1],), name='B_3b%s' % i)
                      for i in range(self.n_weights_3b)]

        self.keep_prob_2b_ph  = tf.placeholder(tf.float32, name='keep_2b')
        self.keep_prob_eam_ph = tf.placeholder(tf.float32, name='keep_eam')
        self.keep_prob_3b_ph  = tf.placeholder(tf.float32, name='keep_3b')
        self.learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')

    def init_weight_variable(self, shape, std=None, name=None):
        if std is None:
            std = self.std_init_weights
        initial = tf.truncated_normal(shape, stddev=std, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def get_architecture(self, layer_nodes, x, Ws, Bs, keep_prob, activation='relu', non_negative=None):
        n_weight_layers = len(layer_nodes)-1
        if isinstance(self.activation, str):
            activation = [self.activation for _ in range(n_weight_layers)]
        Layers = {0: x}
        for i in range(n_weight_layers):
            dense = tf.matmul(Layers[i], Ws[i]) + Bs[i]
            if i != n_weight_layers-1:
                if activation[i] == 'relu':
                    Layers[i+1] = tf.nn.relu(dense)
                elif activation[i] == 'tanh':
                    Layers[i+1] = tf.nn.tanh(dense)
                elif activation[i] == 'square':
                    Layers[i+1] = tf.square(dense)
                elif activation[i] == 'gaus':
                    Layers[i+1] = tf.exp( -0.5 * dense**2 /1.**2)
                elif activation[i] == 'softplus':
                    Layers[i+1] = tf.nn.softplus(dense)
                elif activation[i] == 'sigmoid':
                    Layers[i+1] = tf.nn.sigmoid(dense)
                Layers[i+1] = tf.nn.dropout(Layers[i+1], keep_prob)
            else:
                if non_negative is None:
                    out = dense
                else:
                    # make layer positive by relu or square
                    # using relu gibes however more likely zero weights.
                    relu = tf.square(dense)
                    splits = tf.split(relu, [1]*layer_nodes[-1], axis=1)
                    splits_signed = [s if j in non_negative else -s for j, s in enumerate(splits)]  
                    out = tf.concat(splits_signed, axis=1)
                Layers[i+1] = tf.expand_dims( out, -1)

        return Layers

    def _get_path(self, path=None, filename=None):
        if path is None:
            return self.model_path / filename
        else:
            return path

    def save_nn(self, path=None):
        
        parameters_dict = {'activation' : self.activation,
                           'architecture_2b': self.architecture_2b,
                           'architecture_3b': self.architecture_3b,
                           'architecture_eam': self.architecture_eam,
                           'std_init_weights': self.std_init_weights,
                           'non_negative': self.non_negative,
                           'n_weights_2b': self.n_weights_2b,
                           'n_weights_3b': self.n_weights_3b,
                           'n_weights_eam': self.n_weights_eam,
                           'optimizer_type': self.optimizer_type,
                           'learning_rate': self.learning_rate,
                           'n_epochs': self.n_epochs,
                           'batch_size': self.batch_size,
                           'stopping_threshold': self.stopping_threshold,
                           'lambda_reg': self.lambda_reg,
                           'keep_2b': self.keep_2b,
                           'keep_3b': self.keep_3b,
                           'keep_eam': self.keep_eam,
                           'decay': self.decay,
                           'epoch_save_weights': self.epoch_save_weights,
                           'idx_save_weights': self.idx_save_weights,
                           'epoch': self.epoch,
                           'rmses': self.rmses,
                           'start_3b' : self.start_3b,
                           'fit_first_time': self.fit_first_time,
                           'is_3b_activated': self.is_3b_activated,
                           }
        
        path_json = self._get_path(path=path, filename='nn.json')
        path_tf = str(self._get_path(path=path, filename='nn'))
        json.dump(parameters_dict, open(path_json, 'w'), indent=0)
        saver = tf.train.Saver()
        saver.save(self.sess, path_tf)
    
    def save_weights(self, path=None, filename='weights', unnormalize=False):
        
        Ws_2b  = self.sess.run(self.Ws_2b)
        Ws_eam = self.sess.run(self.Ws_eam)
        Ws_3b  = self.sess.run(self.Ws_3b)
        Bs_2b  = self.sess.run(self.Bs_2b)
        Bs_3b  = self.sess.run(self.Bs_3b)
        Bs_eam = self.sess.run(self.Bs_eam)

        if unnormalize:
            Ws_2b, Ws_eam, Ws_3b, Bs_2b, Bs_eam, Bs_3b = self._get_unnormalized_weights(Ws_2b, Ws_eam, Ws_3b, 
                                                                                        Bs_2b, Bs_eam, Bs_3b)
        
        weights_dict = {'Ws_2b':  Ws_2b,
                        'Ws_eam': Ws_eam,
                        'Ws_3b':  Ws_3b,
                        'Bs_2b':  Bs_2b,
                        'Bs_eam': Bs_eam,
                        'Bs_3b':  Bs_3b,
                        'epoch':  self.epoch
                        }
        
        path = self._get_path(path=path, filename=filename+'.npy')
        np.save(path, weights_dict)

    def load_nn(self, path=None):
        path_json = self._get_path(path=path, filename='nn.json')
        path_tf = str(self._get_path(path=path, filename='nn'))

        self.sess = tf.InteractiveSession()
        parameters_dict = json.load(open(path_json))
        self.set_nn(parameters_dict)
        self.load_tf_sess(filename=path_tf)

    def load_weights(self, path=None, filename='weights'):
        path = self._get_path(path=path, filename=filename+'.npy')
        weights_dict = np.load(path)[()]
        self.set_weights(weights_dict['Ws_2b'], weights_dict['Ws_eam'], weights_dict['Ws_3b'],
                         weights_dict['Bs_2b'], weights_dict['Bs_eam'], weights_dict['Bs_3b'])
        return weights_dict['epoch']
     

    def load_tf_sess(self, filename='nn'):
        self.sess = tf.InteractiveSession()

        saver = tf.train.import_meta_graph(filename + ".meta")
        saver.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(filename)))
        
        graph = tf.get_default_graph()
        
        self.Xs_ph  = [graph.get_tensor_by_name('X%s:0' % i) for i in range(12)]
        self.Gs_ph  = [graph.get_tensor_by_name('G%s:0' % i) for i in range(12)]
        self.Ws_2b  = [graph.get_tensor_by_name('W_2b%s:0' % i) for i in range(self.n_weights_2b)]
        self.Ws_eam = [graph.get_tensor_by_name('W_eam%s:0' % i) for i in range(self.n_weights_eam)]
        self.Ws_3b  = [graph.get_tensor_by_name('W_3b%s:0' % i) for i in range(self.n_weights_3b)]
        self.Bs_2b  = [graph.get_tensor_by_name('B_2b%s:0' % i) for i in range(self.n_weights_2b)]
        self.Bs_eam = [graph.get_tensor_by_name('B_eam%s:0' % i) for i in range(self.n_weights_eam)]
        self.Bs_3b  = [graph.get_tensor_by_name('B_3b%s:0' % i) for i in range(self.n_weights_3b)]
        self.Y_ph             = graph.get_tensor_by_name('Y:0')
        self.keep_prob_2b_ph  = graph.get_tensor_by_name('keep_2b:0')
        self.keep_prob_eam_ph = graph.get_tensor_by_name('keep_eam:0')
        self.keep_prob_3b_ph  = graph.get_tensor_by_name('keep_3b:0')
        self.learning_rate_ph = graph.get_tensor_by_name('learning_rate:0')
        self.loss             = graph.get_tensor_by_name('loss:0')
        self.loss_2b          = graph.get_tensor_by_name('loss_2b:0')
        self.y_model          = graph.get_tensor_by_name('y_model:0')
        self.y_model_2b       = graph.get_tensor_by_name('y_model_2b:0')
        self.y_model_2b_strict= graph.get_tensor_by_name('y_model_2b_strict:0')
        self.coefs_pair       = graph.get_tensor_by_name('coefs_pair:0')
        self.coefs_triplet    = graph.get_tensor_by_name('coefs_triplet:0')
        self.coefs            = graph.get_tensor_by_name('coefs:0')
        self.coefs_2b         = graph.get_tensor_by_name('coefs_2b:0')

        self.train_step    = tf.get_collection("train_step")[0]
        self.train_step_2b = tf.get_collection("train_step_2b")[0]

    def reset_optimizer(self):
        """Checked only for Adam optimizer"""
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,)
        variables_opt = [v for v in variables if self.optimizer_type in v.name.lower()
                         or v.name in ['beta1_power:0', 'beta2_power:0']]
        self.sess.run(tf.variables_initializer(variables_opt))

    def set_nn(self, parameters_dict):
        self.__dict__.update(parameters_dict)
       
    def set_weights(self, Ws_2b, Ws_eam, Ws_3b, Bs_2b, Bs_eam, Bs_3b):
        for i in range(self.n_weights_2b):
            self.sess.run(tf.assign(self.Ws_2b[i], Ws_2b[i]))
            self.sess.run(tf.assign(self.Bs_2b[i], Bs_2b[i]))
        for i in range(self.n_weights_eam):
            self.sess.run(tf.assign(self.Ws_eam[i], Ws_eam[i]))
            self.sess.run(tf.assign(self.Bs_eam[i], Bs_eam[i]))
        for i in range(self.n_weights_3b):
            self.sess.run(tf.assign(self.Ws_3b[i], Ws_3b[i]))
            self.sess.run(tf.assign(self.Bs_3b[i], Bs_3b[i]))

    def init_nn(self, optimizer_type='adam', learning_rate=0.011, n_epochs=200000, 
                stopping_threshold=0., batch_size=None, lambda_reg=0., keep_2b=1., 
                keep_3b=1., keep_eam=1., epoch_save_weights=None, 
                decay=None, epoch_start_3b=-1,):
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.stopping_threshold = stopping_threshold
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.keep_2b = keep_2b
        self.keep_3b = keep_3b
        self.keep_eam = keep_eam
        self.decay = decay
        
        self.epoch_save_weights = epoch_save_weights
        self.start_3b = epoch_start_3b
            
    def set_optimization_problem(self):
        layers_2b  = [self.get_architecture(self.architecture_2b, self.Xs_ph[i], self.Ws_2b, self.Bs_2b, 
                                            self.keep_prob_2b_ph, activation=self.activation, 
                                            non_negative=self.non_negative) 
                                            for i in range(3)]

        layers_eam = [self.get_architecture(self.architecture_eam, self.Xs_ph[i], self.Ws_eam, self.Bs_eam, 
                                            self.keep_prob_eam_ph, activation=self.activation)
                                            for i in range(3, 6)]

        #layers_eam[1][self.n_weights_eam] = layers_eam[1][self.n_weights_eam] * 0.

        layers_3b  = [self.get_architecture(self.architecture_3b, self.Xs_ph[i], self.Ws_3b, self.Bs_3b, 
                                            self.keep_prob_3b_ph, activation=self.activation) 
                                            for i in range(6, 12)]
        
        coefs_list   = [layers_2b[i][self.n_weights_2b] for i in range(3)]
        coefs_list  += [layers_eam[i][self.n_weights_eam] for i in range(3)]
        coefs_list  += [layers_3b[i][self.n_weights_3b] for i in range(6)]

        self.coefs    = tf.concat(coefs_list,     axis=1, name="coefs")  
        self.coefs_2b = tf.concat(coefs_list[:6], axis=1, name="coefs_2b") 
        self.coefs_2b_strict = tf.concat(coefs_list[:3], axis=1, name="coefs_2b_strict") 

        #needed for prediction
        self.coefs_pair    = tf.squeeze(coefs_list[0], name="coefs_pair")
        self.coefs_triplet = tf.squeeze(coefs_list[6], name="coefs_triplet")

        G    = tf.expand_dims(tf.concat(self.Gs_ph,     axis=1), 1)
        G_2b = tf.expand_dims(tf.concat(self.Gs_ph[:6], axis=1), 1)
        G_2b_strict = tf.expand_dims(tf.concat(self.Gs_ph[:3], axis=1), 1)
        
        self.y_model    = tf.squeeze(tf.matmul(G,    self.coefs   ), name="y_model")
        self.y_model_2b = tf.squeeze(tf.matmul(G_2b, self.coefs_2b), name="y_model_2b")
        self.y_model_2b_strict = tf.squeeze(tf.matmul(G_2b_strict, self.coefs_2b_strict), name="y_model_2b_strict")

        self.loss    = tf.reduce_sum(tf.square(self.y_model    - self.Y_ph), name='loss')
        self.loss_2b = tf.reduce_sum(tf.square(self.y_model_2b - self.Y_ph), name='loss_2b')
       
        if self.lambda_reg >0.:
            l2_penalty_2b  = self.lambda_reg * (tf.reduce_sum([tf.nn.l2_loss(w) for w in self.Ws_2b[:]])
                                             +  tf.reduce_sum([tf.nn.l2_loss(b) for b in self.Bs_2b[:]]))
            l2_penalty_eam = self.lambda_reg * (tf.reduce_sum([tf.nn.l2_loss(w) for w in self.Ws_eam[:]])
                                             +  tf.reduce_sum([tf.nn.l2_loss(b) for b in self.Bs_eam[:]]))
            l2_penalty_3b  = self.lambda_reg * (tf.reduce_sum([tf.nn.l2_loss(w) for w in self.Ws_3b[:]]) 
                                             +  tf.reduce_sum([tf.nn.l2_loss(b) for b in self.Bs_3b[:]]))

            self.loss    += self.lambda_reg * l2_penalty_3b
            #self.loss    += self.lambda_reg * (l2_penalty_2b + l2_penalty_eam + l2_penalty_3b)
            self.loss_2b += self.lambda_reg * (l2_penalty_2b + l2_penalty_eam)
    
    def set_optimizer(self):
        self.sess = tf.InteractiveSession()

        if self.optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        elif self.optimizer_type == 'gradient_descent':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_ph)
        elif self.optimizer_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate_ph)
        elif self.optimizer_type == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate_ph)


        self.train_step    = self.optimizer.minimize(self.loss)
        self.train_step_2b = self.optimizer.minimize(self.loss_2b)
        tf.global_variables_initializer().run()
        tf.add_to_collection("train_step",    self.train_step)
        tf.add_to_collection("train_step_2b", self.train_step_2b)
    
    def _get_normalized_data(self, Xs, Gs, fit=True):
        Xs = [X.copy() for X in Xs]
        Gs = [G.copy() for G in Gs]
        if fit:
            self.standardizer_2b  = Standardizer()
            self.standardizer_eam = Standardizer()
            self.standardizer_3b  = Standardizer()
            self.normalizer_2b    = Normalizer()
            self.normalizer_eam   = Normalizer()
            self.normalizer_3b    = Normalizer()

            self.standardizer_2b.fit(np.vstack(Xs[:3]))
            self.standardizer_eam.fit(np.vstack(Xs[3:6]))
            self.standardizer_3b.fit(np.vstack(Xs[6:]))
            self.normalizer_2b.fit(np.vstack(Gs[:3]))
            self.normalizer_eam.fit(np.vstack(Gs[3:6]))
            self.normalizer_3b.fit(np.vstack(Gs[6:]))
        for i in range(12):
            if i < 3:
                Xs[i] = self.standardizer_2b.transform(Xs[i])
                Gs[i] = self.normalizer_2b.transform(Gs[i])
                pot_weight = self.pot_weights[0]
            elif i < 6:
                Xs[i] = self.standardizer_eam.transform(Xs[i])
                Gs[i] = self.normalizer_eam.transform(Gs[i])
                pot_weight = self.pot_weights[1]
            else:
                Xs[i] = self.standardizer_3b.transform(Xs[i])
                Gs[i] = self.normalizer_3b.transform(Gs[i]) 
                pot_weight = self.pot_weights[2]
            Gs[i] *= pot_weight
        return Xs, Gs

    def _get_unnormalized_weights(self,  Ws_2b, Ws_eam, Ws_3b, Bs_2b, Bs_eam, Bs_3b):
        
        Ws_2b[0], Bs_2b[0] = self.standardizer_2b.invert_parameters(Ws_2b[0], Bs_2b[0])
        Ws_2b[-1] = self.normalizer_2b.invert_parameters(Ws_2b[-1], non_negative=self.non_negative) * self.pot_weights[0]
        Bs_2b[-1] = self.normalizer_2b.invert_parameters(Bs_2b[-1], non_negative=self.non_negative) * self.pot_weights[0]


        Ws_eam[0], Bs_eam[0] = self.standardizer_eam.invert_parameters(Ws_eam[0], Bs_eam[0])
        Ws_eam[-1] = self.normalizer_eam.invert_parameters(Ws_eam[-1]) * self.pot_weights[1]
        Bs_eam[-1] = self.normalizer_eam.invert_parameters(Bs_eam[-1]) * self.pot_weights[1]
        
        if self.is_3b_activated:
            Ws_3b[0], Bs_3b[0] = self.standardizer_3b.invert_parameters(Ws_3b[0], Bs_3b[0])
            Ws_3b[-1] = self.normalizer_3b.invert_parameters(Ws_3b[-1]) * self.pot_weights[2]
            Bs_3b[-1] = self.normalizer_3b.invert_parameters(Bs_3b[-1]) * self.pot_weights[2]

        return  Ws_2b, Ws_eam, Ws_3b, Bs_2b, Bs_eam, Bs_3b

    def _get_normalized_weights(self,  Ws_2b, Ws_eam, Ws_3b, Bs_2b, Bs_eam, Bs_3b):
        
        Ws_2b[0], Bs_2b[0] = self.standardizer_2b.transform_parameters(Ws_2b[0], Bs_2b[0])
        Ws_2b[-1] = self.normalizer_2b.transform_parameters(Ws_2b[-1], non_negative=self.non_negative) / self.pot_weights[0]
        Bs_2b[-1] = self.normalizer_2b.transform_parameters(Bs_2b[-1], non_negative=self.non_negative) / self.pot_weights[0]
        
        Ws_eam[0], Bs_eam[0] = self.standardizer_eam.transform_parameters(Ws_eam[0], Bs_eam[0])
        Ws_eam[-1] = self.normalizer_eam.transform_parameters(Ws_eam[-1]) / self.pot_weights[1]
        Bs_eam[-1] = self.normalizer_eam.transform_parameters(Bs_eam[-1]) / self.pot_weights[1]

        if self.is_3b_activated:
            Ws_3b[0], Bs_3b[0] = self.standardizer_3b.transform_parameters(Ws_3b[0], Bs_3b[0])
            Ws_3b[-1] = self.normalizer_3b.transform_parameters(Ws_3b[-1]) / self.pot_weights[2]
            Bs_3b[-1] = self.normalizer_3b.transform_parameters(Bs_3b[-1]) / self.pot_weights[2]

        return  Ws_2b, Ws_eam, Ws_3b, Bs_2b, Bs_eam, Bs_3b
    
    def _set_weights_normalization(self, unnormalize=True):
        Ws_2b  = self.sess.run(self.Ws_2b)
        Ws_eam = self.sess.run(self.Ws_eam)
        Ws_3b  = self.sess.run(self.Ws_3b)
        Bs_2b  = self.sess.run(self.Bs_2b)
        Bs_eam = self.sess.run(self.Bs_eam)
        Bs_3b  = self.sess.run(self.Bs_3b)
        
        if unnormalize:
            weights = self._get_unnormalized_weights(Ws_2b, Ws_eam, Ws_3b, Bs_2b, Bs_eam, Bs_3b)
        else:
            weights = self._get_normalized_weights(Ws_2b, Ws_eam, Ws_3b, Bs_2b, Bs_eam, Bs_3b)

        self.set_weights(*weights)
    
    def _check_if_first_time(self):
        if self.fit_first_time:
            self.init_tensors()
            self.set_optimization_problem()
            self.set_optimizer()
            self.fit_first_time = False
            self.idx_save_weights = 0
            self.epoch = 0
            self.rmses = []
        else:
            self._set_weights_normalization(unnormalize=False)
    
    def fit(self, Xs, Gs, Y, vali=None):
        pathlib.Path(self.model_path).mkdir(parents=True, exist_ok=True)
        
        Xs, Gs = self._get_normalized_data(Xs, Gs)
        self._check_if_first_time()
       
        
        n_samples = len(Y)
        dic_train_set = {**dict(zip(self.Gs_ph, Gs)),
                         **dict(zip(self.Xs_ph, Xs)),
                         self.Y_ph: Y, 
                         self.keep_prob_2b_ph: 1.0,
                         self.keep_prob_eam_ph: 1.0,
                         self.keep_prob_3b_ph: 1.0
                         }
        if vali is not None:
            Xs_vali, Gs_vali, Y_vali = vali
            Xs_vali, Gs_vali = self._get_normalized_data(Xs_vali, Gs_vali, fit=False)
            n_samples_vali = len(Y_vali)
            dic_vali_set  = {**dict(zip(self.Gs_ph, Gs_vali)),
                             **dict(zip(self.Xs_ph, Xs_vali)),
                             self.Y_ph: Y_vali, 
                             self.keep_prob_2b_ph: 1.0,
                             self.keep_prob_eam_ph: 1.0,
                             self.keep_prob_3b_ph: 1.0
                             }
            old_rmse_vali = np.inf

        if self.batch_size is None:
            batch_size = n_samples
        else:
            batch_size = self.batch_size
        n_batches_total = n_samples // batch_size + 1
        indices_for_batch = list(range(n_samples))

        ######### TRAIN
        print(self.epoch-1, round(np.sqrt(self.sess.run(self.loss, dic_train_set)/n_samples), 4))
        t1 = time()
        for epoch_here in range(self.n_epochs):
            if self.decay is not None and self.epoch == self.decay[0]:
                print("Old learning rate:", self.learning_rate)
                self.learning_rate *= self.decay[1]
                print("Old learning rate:", self.learning_rate)
            # batch learning
            shuffle(indices_for_batch)
            for i_batch in range(n_batches_total):
                batch_indices = indices_for_batch[ i_batch * batch_size: (i_batch + 1) * batch_size]
                Gs_batch = [Gs[i][batch_indices] for i in range(12)]
                Xs_batch = [Xs[i][batch_indices] for i in range(12)]
                Y_batch  = Y[batch_indices] 
                
                dic_batch = {**dict(zip(self.Gs_ph, Gs_batch)),
                             **dict(zip(self.Xs_ph, Xs_batch)),
                             self.Y_ph: Y_batch, self.keep_prob_2b_ph:self.keep_2b,
                             self.keep_prob_eam_ph:self.keep_eam,
                             self.keep_prob_3b_ph:self.keep_3b,
                             self.learning_rate_ph: self.learning_rate
                             }

                if self.epoch < self.start_3b:
                    self.sess.run(self.train_step_2b, dic_batch)
                else:
                    self.sess.run(self.train_step, dic_batch)
                    

            # print current loss
            if self.epoch < self.start_3b:
                loss = self.loss_2b
            else:
                self.is_3b_activated = True
                loss = self.loss
            
            curr_rmse = np.sqrt(self.sess.run(loss, dic_train_set) / n_samples)
            
            if vali is not None:
                curr_rmse_vali = np.sqrt(self.sess.run(loss, dic_vali_set) / n_samples_vali)
                print("%s  %.4f  %.4f" % (self.epoch, curr_rmse, curr_rmse_vali))

                if curr_rmse_vali < old_rmse_vali:
                    self.save_weights(filename='weights_best', unnormalize=True)
                    old_rmse_vali = curr_rmse_vali
            else:
                print(self.epoch, round(curr_rmse, 4))
            
            if self.epoch_save_weights is not None and epoch_here % self.epoch_save_weights == 0 and epoch_here > 0:
                self.save_weights(filename='weights_%s' % self.idx_save_weights, unnormalize=True)
                self.idx_save_weights += 1
            
            if curr_rmse < self.stopping_threshold:
                break

            self.epoch += 1
        
        self._set_weights_normalization(unnormalize=True)
        print('Training time:', time()-t1)
        print("######### TRAINING FINISHED")

    def predict(self, Xs, Gs):

        dic = {**{self.Gs_ph[i]: Gs[i] for i in range(12)},
               **{self.Xs_ph[i]: Xs[i] for i in range(12)},
               self.keep_prob_2b_ph:1.0,
               self.keep_prob_eam_ph:1.0,
               self.keep_prob_3b_ph:1.0
               }
        if self.epoch < self.start_3b:
            return self.sess.run(self.y_model_2b, dic)       
        else:
            return self.sess.run(self.y_model, dic)       

    def predict_2b(self, Xs, Gs):

        dic = {**{self.Gs_ph[i]: Gs[i] for i in range(3)},
               **{self.Xs_ph[i]: Xs[i] for i in range(3)},
               self.keep_prob_2b_ph:1.0,
               self.keep_prob_eam_ph:1.0,
               self.keep_prob_3b_ph:1.0
               }
        return self.sess.run(self.y_model_2b_strict, dic)       
    
    def predict_2beam(self, Xs, Gs):

        dic = {**{self.Gs_ph[i]: Gs[i] for i in range(12)},
               **{self.Xs_ph[i]: Xs[i] for i in range(12)},
               self.keep_prob_2b_ph:1.0,
               self.keep_prob_eam_ph:1.0,
               self.keep_prob_3b_ph:1.0
               }
        return self.sess.run(self.y_model_2b, dic)       

    def get_coefs(self, Xs):
        dic = {**{self.Xs_ph[i]: Xs[i] for i in range(12)}, 
               self.keep_prob_2b_ph:1.0, 
               self.keep_prob_eam_ph:1.0,
               self.keep_prob_3b_ph:1.0}
        if self.is_3b_activated:
            return self.sess.run(self.coefs, dic)
        else:
            return self.sess.run(self.coefs_2b, dic)

    def get_coefs_2b(self, Xs):
        dic = {**{self.Xs_ph[i]: Xs[i] for i in range(12)}, 
               self.keep_prob_2b_ph:1.0, 
               self.keep_prob_eam_ph:1.0,
               self.keep_prob_3b_ph:1.0}
        return self.sess.run(self.coefs_2b, dic)

    def get_coefs_pair(self, X):
        dic = {self.Xs_ph[0]: X, self.keep_prob_2b_ph:1.0, 
               self.keep_prob_eam_ph:1.0,
               self.keep_prob_3b_ph:1.0}
        return self.sess.run(self.coefs_pair, dic)

    def get_coefs_triplet(self, X):
        dic = {self.Xs_ph[6]: X, self.keep_prob_2b_ph:1.0, 
                                 self.keep_prob_eam_ph:1.0,
                                 self.keep_prob_3b_ph:1.0}
        return self.sess.run(self.coefs_triplet, dic)

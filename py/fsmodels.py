"""Модуль, содержащий модели частотных сканов, полученных на DLS-82E.

"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf



class BaseModel(BaseEstimator, RegressorMixin):
    
    
    def __init__(self,
                 filling_pulse = 20*10**-6,
                 n_exps = 1,
                 learning_rate = 0.1,
                 n_iters = 1000,
                 stop_val = None,
                 verbose = False
                ):
        self.filling_pulse = filling_pulse
        self.n_exps = n_exps
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.stop_val = stop_val
        self.verbose = verbose

        
    def _get_phi(self,
                 frequency_powers,
                 time_constant_power,
                ):
        time_constant = tf.pow(10.0, time_constant_power)
        frequency = tf.pow(10.0, frequency_powers)

        a = time_constant * frequency
        b = self.filling_pulse * frequency

        exp0 = tf.exp(-0.05 / (a))
        exp1 = tf.exp((b - 0.45) / (a))
        exp2 = tf.exp(-0.5 / (a))
        exp3 = tf.exp((b - 0.95) / (a))

        return a * exp0 * (1.0 - exp1 - exp2 + exp3)


    def _get_M(self,
               time_constant_power,
               learning_rate=0.2, 
               n_iters=100, 
               stop_val = 10**-10
              ):
        prev_loss = tf.Variable(np.inf, dtype='float64')
        max_freq_pow = tf.Variable(-time_constant_power, dtype='float64')

        for _ in range(n_iters):
            with tf.GradientTape() as tape:

                current_loss = 0.0 - self._get_phi(max_freq_pow, time_constant_power)

            if stop_val is not None:
                if tf.abs(current_loss - prev_loss) < stop_val:
                    break

            dfreq_pow = tape.gradient(current_loss, max_freq_pow)
            max_freq_pow.assign_sub(learning_rate * dfreq_pow)

            prev_loss = current_loss

        return tf.Variable([-1 / current_loss])
    
    
    def predict(self, X):
        f_powers = tf.Variable(X, dtype='float64')
        return self._get_dlts(X=f_powers).numpy()
    
    
    def _get_dlts(self, X):
        raise NotImplementedError('Implement _get_dlts() in ' + self.__class__.__name__ + '.') 
        
        
    def fit(self, X, y, initial_exps_params_=None):
        raise NotImplementedError('Implement fit() in ' + self.__class__.__name__ + '.') 
    
    
    def _update_M(self):
        self._M = tf.map_fn(fn = self._get_M, elems=self._exps_params[:, 0])
    
    
    @property
    def exps_params_(self):
        '''exps_params_ = [[timeconstant_power_0, amplitude_0],
                           [timeconstant_power_1, amplitude_1],
                           ... ,
                           [timeconstant_power_n, amplitude_n]]'''
        return self._exps_params.numpy()
    
    @exps_params_.setter
    def exps_params_(self, val):
        value = tf.Variable(val, dtype='float64')
        
        condition = not(len(value.shape) == 2 and 
                        value.shape[0] == self.n_exps and 
                        value.shape[1] == 2)
        
        if condition:
            raise ValueError('The shape of exps_params must be equal to [n_exps, 2], ' + 
                             f'specifically {[self.n_exps, 2]}.')
        self._exps_params = value
        self._update_M()
        
        
    @property
    def fit_results_(self):
        return self._fit_results
    
    
    def _get_fit_result(self, loss=None):
        fit_result = pd.DataFrame(self.get_params(), index=[0])
        fit_result['n_exps'] = self.n_exps
        fit_result['loss'] = loss.numpy()
        
        for i, exp_param in enumerate(self.exps_params_):
            fit_result[f'time_constant_pow_{i}'] = exp_param[0]
            fit_result[f'amplitude_{i}'] = exp_param[1]
            
        try:
            fit_result['p_coef'] = self.p_coef_
        except AttributeError:
            pass
        
        return fit_result.sort_index(axis='columns')
        
    
    def print_all_params(self, iteration_number=None, loss=None):
        if iteration_number is not None:
            print(f'iteration # {iteration_number}')
            
        if loss is not None:
            print(f'loss: {loss}')
            
        print(f'exps_params:\n{self.exps_params_}')
        
        try:
            print(f'p_coef: {self.p_coef_}')
        except AttributeError:
            pass
        
        params = self.get_params()
        for key in params.keys():
            print(f'{key}: {params[key]}')
            
        print('\n')
   

        
class SklSingleExpFrequencyScan(BaseModel):
    
    
    def __init__(self,
                 filling_pulse = 20*10**-6,
                 fit_p_coef = True,
                 learning_rate = 0.1,
                 n_iters = 1000,
                 stop_val = None,
                 verbose = False
                ):
        
        super().__init__(filling_pulse = filling_pulse,
                         n_exps = 1,
                         learning_rate = learning_rate,
                         n_iters = n_iters,
                         stop_val = stop_val,
                         verbose = verbose)
        
        self.fit_p_coef = fit_p_coef
        
        
    def _get_dlts(self, X):
        frequency_powers = tf.Variable(X, dtype='float64')
        
        phi = self._get_phi(frequency_powers, self._exps_params[0, 0])
        
        return self._exps_params[0, 1] * tf.pow(self._M[0] * phi, self._p_coef)
    
    
    @property
    def p_coef_(self):
        return self._p_coef.numpy()

    @p_coef_.setter
    def p_coef_(self, val):
        self._p_coef = tf.Variable(val, dtype='float64')
        
        
    def fit(self, X, y, initial_exps_params_=None):
        
        if initial_exps_params_ is None:
            self.exps_params_ = [[np.random.uniform(low=-3.5, high=-1), np.random.uniform(low=-1, high=1)]]
        else:
            self.exps_params_ = initial_exps_params_

        self._update_M()
        self.p_coef_ = 1.0
        
        frequency_powers = tf.Variable(X, dtype='float64')
        dlts = tf.Variable(y, dtype='float64')
        
        prev_loss = tf.Variable(np.inf, dtype='float64')
        
        self._fit_results = pd.DataFrame()
        
        for i in range(self.n_iters):
            with tf.GradientTape() as tape:
                predicted_dlts = self._get_dlts(frequency_powers)
                current_loss = tf.reduce_mean(tf.square(dlts - predicted_dlts))
                
            if self.fit_p_coef:
                d_exps_params, d_p_coef = tape.gradient(current_loss, [self._exps_params, self._p_coef])
            else:
                d_exps_params = tape.gradient(current_loss, self._exps_params)
                
            self._fit_results = pd.concat([self._fit_results, 
                                           self._get_fit_result(loss=current_loss)
                                          ],
                                          ignore_index=True)
            if self.verbose:
                self.print_all_params(iteration_number=i, loss=current_loss)
                
            if self.stop_val is not None:
                if tf.abs(current_loss - prev_loss) < self.stop_val:
                    break
                
            self._exps_params.assign_sub(self.learning_rate * d_exps_params)
            self._update_M()
            
            if self.fit_p_coef:
                self._p_coef.assign_sub(self.learning_rate * d_p_coef)
                
            prev_loss = current_loss
        
        return self
    
    
    
class SklMultiExpFrequencyScan(BaseModel):
    
    
    def __init__(self,
                 filling_pulse = 20*10**-6,
                 n_exps = 1,
                 learning_rate = 0.1,
                 n_iters = 1000,
                 stop_val = None,
                 verbose = False
                ):
        
        super().__init__(filling_pulse = filling_pulse,
                         n_exps = n_exps,
                         learning_rate = learning_rate,
                         n_iters = n_iters,
                         stop_val = stop_val,
                         verbose = verbose)
        
    
    def _get_dlts(self, X):
        frequency_powers = tf.Variable(X, dtype='float64')
        
        def get_one_term(params):
            exp_params = params[:2]
            M = params[2]
        
            phi = self._get_phi(frequency_powers, exp_params[0])

            return exp_params[1] * M * phi
        
        params = tf.concat([self._exps_params, self._M], axis=1)
        terms = tf.map_fn(fn=get_one_term, 
                          elems=params, 
                          fn_output_signature=tf.float64
                         )
        
        return tf.reduce_sum(terms, axis=0)
    
    
    def fit(self, X, y, initial_exps_params_=None):
        
        
        frequency_powers = tf.Variable(X, dtype='float64')
        dlts = tf.Variable(y, dtype='float64')
        
        self._fit_results = pd.DataFrame()
        prev_loss = tf.Variable(np.inf, dtype='float64')
        
        if initial_exps_params_ is None:
            self.exps_params_ = [[np.random.uniform(low=-3.5, high=-1), 
                                  np.random.uniform(low=-1/self.n_exps, high=1/self.n_exps)] 
                                 for _ in range(self.n_exps)]
        else:
            self.exps_params_ = initial_exps_params_

        self._update_M()
        
        for i in range(self.n_iters):
            with tf.GradientTape() as tape:
                predicted_dlts = self._get_dlts(frequency_powers)
                current_loss = tf.reduce_mean(tf.square(dlts - predicted_dlts))
                
            d_exps_params = tape.gradient(current_loss, self._exps_params)
            
            self._fit_results = pd.concat([self._fit_results,
                                           self._get_fit_result(loss=current_loss)
                                          ],
                                          ignore_index = True)
            
            if self.verbose:
                self.print_all_params(iteration_number=i, loss=current_loss)
                
            if self.stop_val is not None:
                if tf.abs(current_loss - prev_loss) < self.stop_val:
                    break
                    
            self._exps_params.assign_sub(self.learning_rate * d_exps_params)
            self._update_M()
        
            prev_loss = current_loss
 
        return self
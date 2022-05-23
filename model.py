import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class MonoexponentialModel(tf.keras.Model):
    def __init__(self, 
                 filling_pulse=20 * 10 ** (-6), 
                 time_constant_power=None,
                 amplitude=None,
                 M=5.861, 
                 **kwargs):
        
        super().__init__(**kwargs)            
        
        self.filling_pulse = tf.Variable(filling_pulse, trainable=False)
        self.M = tf.Variable(M, trainable=False)
        
        if time_constant_power is None:
            power = float(np.random.uniform(low=-3, high=-0.5))
            self.time_constant_power = tf.Variable(power)
        else:
            self.time_constant_power = tf.Variable(time_constant_power)
            
        if amplitude is None:
            amp = float(np.random.uniform(low=0.3, high=3.3))
            self.amplitude = tf.Variable(amp)
        else:
            self.amplitude = tf.Variable(amplitude)
        
        
    def get_phi(self, frequency):
        time_constant = tf.pow(10.0, self.time_constant_power)
        
        exp0 = tf.exp(-0.05 / (time_constant * frequency))
        exp1 = tf.exp((self.filling_pulse * frequency - 0.45) / (time_constant * frequency))
        exp2 = tf.exp(-0.5 / (time_constant * frequency))
        exp3 = tf.exp((self.filling_pulse * frequency - 0.95) / (time_constant * frequency))
        
        phi = time_constant * frequency * exp0 * (1.0 - exp1 - exp2 + exp3)
        
        return phi
      
    def get_max_phi(self):
        f_pwr_grid = tf.linspace(start=tf.experimental.numpy.log10(1.0), 
                         stop=tf.experimental.numpy.log10(2500.0), 
                         num=100000, 
                         name=None, 
                         axis=0)
        
        f_grid = tf.pow(tf.constant(10.0, dtype='float64'), f_pwr_grid)
        f_grid = tf.cast(f_grid, dtype='float32')
        
        M = 1.0 / tf.math.reduce_max(self.get_phi(f_grid))
        
        return M
        
    
    def call(self, frequency):
        
        self.M = self.get_max_phi()
        
        dlts = self.amplitude * self.M * self.get_phi(frequency)
        
        return dlts
    
    
    
class MonoexponentialModelP(MonoexponentialModel):
    def __init__(self, 
                 filling_pulse=20 * 10 ** (-6), 
                 time_constant_power=None,
                 amplitude=None,
                 p = 1.0,
                 M=5.861, 
                 **kwargs):
        
        super().__init__(filling_pulse=filling_pulse,
                         time_constant_power=time_constant_power,
                         amplitude=amplitude,
                         M=M, 
                         **kwargs)
        
        self.p_coef = tf.Variable(p)
      
    
    def call(self, frequency):
        phi = super().get_phi(frequency)
        
        self.M = self.get_max_phi()
        
        dlts = self.amplitude * ((self.M * phi) ** self.p_coef)
        
        return dlts
    
    
    
def make_exp_data(f_pulse,
                  time_constant,
                  ampl,
                  std_dev,
                  p=1.0,
                  start_f=1, 
                  stop_f=2500,
                  num_ex=1000):
    
    powers = tf.linspace(np.log10(start_f), np.log10(stop_f), num_ex)
    frequency = tf.pow(10.0, powers)
    
    t_c_pwr = float(np.log10(time_constant))

    
    if p == 1.0:
        fs_model = MonoexponentialModel(filling_pulse=f_pulse,
                                        time_constant_power=t_c_pwr,
                                        amplitude=ampl
                                       )
    else:
        fs_model = MonoexponentialModelP(filling_pulse=f_pulse,
                                         time_constant_power=t_c_pwr,
                                         amplitude=ampl,
                                         p=p
                                        )
    

    noise = tf.random.normal(stddev=std_dev, shape=[frequency.shape[0]])
    
    actual_dlts = fs_model(frequency) + noise
    
    return frequency, actual_dlts



def print_results(frequency, actual_dlts, initial_model, history = None, final_model=None):
    print('Initial values:')
    
    time_constant_power = initial_model.time_constant_power.numpy()
    print(f'Time constant power = {time_constant_power:.4f} log10(s)')
    print(f'Time constant = {10**time_constant_power:.4f} s')
    
    print(f'Amplitude = {initial_model.amplitude.numpy():.4f} pF')
    
    init_mse = tf.keras.metrics.mean_squared_error(actual_dlts, initial_model(frequency))
    print(f'MSE = {init_mse:.4f}')
    print(f'RMSE = {tf.sqrt(init_mse):.6f}')
    
    if not (final_model is None):
        print('\nFinal values:')
        
        time_constant_power = final_model.time_constant_power.numpy()
        print(f'Time constant power = {time_constant_power:.4f} log10(s)')
        print(f'Time constant = {10**time_constant_power:.4f} s')
        
        print(f'Amplitude = {final_model.amplitude.numpy():.4f} pF')
        
        final_mse = tf.keras.metrics.mean_squared_error(actual_dlts, final_model(frequency))
        print(f'MSE = {final_mse:.4f}')
        print(f'RMSE = {tf.sqrt(final_mse):.6f}')
        
    
    if not ((final_model is None) or (history is None)):
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(7,5))
        
    ax0.semilogx(frequency, actual_dlts, '.g', label="Actual values", alpha=0.3)
    ax0.semilogx(frequency, initial_model(frequency), '-.b', label="Initial model", alpha=0.5)
    if not (final_model is None):
        ax0.semilogx(frequency, final_model(frequency), 'r', label="Final model")
    
    ax0.set_xlabel('Frequency, Hz')
    ax0.set_ylabel('DLTS')
    ax0.legend()
    ax0.grid()

    if not ((final_model is None) or (history is None)):
        ax1.plot(history.history['loss'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss [Mean Squared Error]')
        plt.ylim([0, max(plt.ylim())])
        ax1.grid()
    
    if not ((final_model is None) or (history is None)):
        return fig, (ax0, ax1)
    else:
        return fig, ax0
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class MonoexponentialModel(tf.keras.Model):
    def __init__(self, 
                 filling_pulse=20 * 10 ** (-6), 
                 time_constant_power=-2.0,
                 amplitude=1.0,
                 M=5.861, 
                 **kwargs):
        
        super().__init__(**kwargs)            
        
        self.filling_pulse = tf.Variable(filling_pulse, trainable=False)
        self.M = tf.Variable(M, trainable=False)
        
        self.time_constant_power = tf.Variable(time_constant_power)
        self.amplitude = tf.Variable(amplitude)
        
    def call(self, frequency):
        
        time_constant = tf.pow(10.0, self.time_constant_power)
        
        exp0 = tf.exp(-0.05 / (time_constant * frequency))
        exp1 = tf.exp((self.filling_pulse * frequency - 0.45) / (time_constant * frequency))
        exp2 = tf.exp(-0.5 / (time_constant * frequency))
        exp3 = tf.exp((self.filling_pulse * frequency - 0.95) / (time_constant * frequency))
        
        phi = time_constant * frequency * exp0 * (1.0 - exp1 - exp2 + exp3)
        
        dlts = self.amplitude * self.M * phi
        
        return dlts
    
    
    
def make_exp_data(f_pulse,
                  time_constant,
                  ampl,
                  std_dev,
                  start_f=1, 
                  stop_f=2500,
                  num_ex=1000):
    
    powers = tf.linspace(np.log10(start_f), np.log10(stop_f), num_ex)
    frequency = tf.pow(10.0, powers)
    
    t_c_pwr = float(np.log10(time_constant))

    fs_model = MonoexponentialModel(filling_pulse=f_pulse,
                                    time_constant_power=t_c_pwr,
                                    amplitude=ampl
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
        

    plt.semilogx(frequency, actual_dlts, '.g', label="Actual values", alpha=0.3)
    plt.semilogx(frequency, initial_model(frequency), '-.b', label="Initial model", alpha=0.5)
    if not (final_model is None):
        plt.semilogx(frequency, final_model(frequency), 'r', label="Final model")
    plt.legend()
    plt.show()

    if not ((final_model is None) or (history is None)):
        plt.plot(history.history['loss'])
        plt.xlabel('Epoch')
        plt.ylim([0, max(plt.ylim())])
        plt.ylabel('Loss [Mean Squared Error]')
        plt.title('Keras training progress')
        plt.show()
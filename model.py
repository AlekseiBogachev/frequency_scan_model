import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
        
        print(f'Amplitude = {final_model.amplitude.numpy():.4f} pf')
        
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
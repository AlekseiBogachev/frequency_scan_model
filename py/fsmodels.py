"""Модуль, содержащий модели частотных сканов, полученных на DLS-82E.
"""

class FrequencyScan(tf.Module):
    """Модель моноэкспоненциального частотного скана с показателем p.

    Модель моноэкспоненциального частотного скана с учётом показателя
    нелинейности-неэкспоненциальности p. Модель позволяет вычислять 
    частотный скан по заданным параметрам, а также идентифицировать 
    параметры модели частотного скана по экспериментальным данным. 
    Термин "моноэкспоненциальный" подразумевает, что модель имеет только
    один набор параметров сигнала релаксации (амплитуда и частота).
    """
    
    def __init__(self,
                 amplitude = 3.5,
                 time_constant_power = -2.0,
                 filling_pulse = 20*10**-6,
                 p_coef = 1.0,
                 
                 fit_p_coef = True,
                 learning_rate = 0.1,
                 n_iters = 1000,
                 stop_val = None,
                 verbose = False,

                 tf_in_out = False,
                 
                 **kwargs
                ):
        super().__init__(**kwargs)
        
        self._amplitude = tf.Variable(amplitude, dtype='float64')
        self._time_constant_power = tf.Variable(time_constant_power, 
                                               dtype='float64')
        self._filling_pulse = tf.Variable(filling_pulse, dtype='float64')
        self._p_coef = tf.Variable(p_coef, dtype='float64')
        
        self.fit_p_coef = fit_p_coef
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.stop_val = stop_val
        self.verbose = verbose

        self.tf_in_out = tf_in_out


    @property
    def amplitude(self):
        if self.tf_in_out:
            return self._amplitude
        else:
            return self._amplitude.numpy()


    @amplitude.setter
    def amplitude(self, a):
        if self.tf_in_out:
            self._amplitude = a
        else:
            self._amplitude = tf.Variable(a, dtype='float64')


    @property
    def time_constant_power(self):
        if self.tf_in_out:
            return self._time_constant_power
        else:
            return self._time_constant_power.numpy()


    @time_constant_power.setter
    def time_constant_power(self, t):
        if self.tf_in_out:
            self._time_constant_power = t
        else:
            self._time_constant_power = tf.Variable(t, dtype='float64')


    @property
    def filling_pulse(self):
        if self.tf_in_out:
            return self._filling_pulse
        else:
            return self._filling_pulse.numpy()


    @filling_pulse.setter
    def filling_pulse(self, f):
        if self.tf_in_out:
            self._filling_pulse = f 
        else:
            self._filling_pulse = tf.Variable(f, dtype='float64')


    @property
    def p_coef(self):
        if self.tf_in_out:
            return self._p_coef
        else:
            return self._p_coef.numpy()


    @p_coef.setter
    def p_coef(self, p):
        if self.tf_in_out:
            self._p_coef = p
        else:
            self._p_coef = tf.Variable(p, dtype='float64')


    def _get_phi(self,
                 frequency_powers
                ):

        time_constant = tf.pow(10.0, self._time_constant_power)
        frequency = tf.pow(10.0, frequency_powers)

        a = time_constant * frequency
        b = self._filling_pulse * frequency

        exp0 = tf.exp(-0.05 / (a))
        exp1 = tf.exp((b - 0.45) / (a))
        exp2 = tf.exp(-0.5 / (a))
        exp3 = tf.exp((b - 0.95) / (a))

        return a * exp0 * (1.0 - exp1 - exp2 + exp3)


    def _get_M(self,
               learning_rate=0.1, 
               n_iters=100,
               stop_val = None,
              ):

        prev_loss = tf.Variable(np.inf, dtype='float64')
        max_freq_pow = tf.Variable(-self._time_constant_power, dtype='float64')

        for _ in range(n_iters):
            with tf.GradientTape() as tape:

                current_loss = 0.0 - self._get_phi(max_freq_pow)

            if stop_val is not None:
                if tf.abs(current_loss - prev_loss) < stop_val:
                    break

            dfreq_pow = tape.gradient(current_loss, max_freq_pow)
            max_freq_pow.assign_sub(learning_rate * dfreq_pow)

            prev_loss = current_loss

        return 1 / self._get_phi(max_freq_pow)
        
        
    def __call__(self, frequency_powers):
        
        M = self._get_M(learning_rate=0.2,
                       n_iters=100,
                       stop_val = 10**-10)
        
        phi = self._get_phi(frequency_powers)
        
        return self._amplitude * tf.pow(M * phi, self._p_coef)
    
    
    def fit(self,
            frequency_powers,
            dlts,
           ):
        
        prev_loss = tf.Variable(np.inf, dtype='float64')
        
        fit_results = pd.DataFrame(columns=['amplitude', 
                                            'time_constant_power', 
                                            'p_coef', 
                                            'loss'])
        
        for _ in range(self.n_iters):
            with tf.GradientTape() as tape:
                predicted_dlts = self.__call__(frequency_powers)
                current_loss = tf.reduce_mean(tf.square(dlts - predicted_dlts))
                
            if self.fit_p_coef:
                dampl, dtime_const_pow, dp_coef = tape.gradient(current_loss,
                    [self._amplitude, self._time_constant_power, self._p_coef])
            else:
                dampl, dtime_const_pow = tape.gradient(current_loss, 
                    [self._amplitude, self._time_constant_power])
                
            fit_results.loc[_, 'amplitude'] = self._amplitude.numpy()
            fit_results.loc[_, 'time_constant_power'] = self._time_constant_power.numpy()
            fit_results.loc[_, 'p_coef'] = self._p_coef.numpy()
            fit_results.loc[_, 'loss'] = current_loss.numpy()     
            
            if self.verbose:
                print('iter #', _)
                print('amp:',self._amplitude)
                print('tau:',self._time_constant_power)
                print('p:', self._p_coef)
                print('Loss:', current_loss)
                
            self._amplitude.assign_sub(self.learning_rate * dampl)
            self._time_constant_power.assign_sub(self.learning_rate * dtime_const_pow)
            if self.fit_p_coef:
                self._p_coef.assign_sub(self.learning_rate * dp_coef)
                
            if self.stop_val is not None:
                if tf.abs(current_loss - prev_loss) < self.stop_val:
                    break
                    
            prev_loss = current_loss
            
        return fit_results
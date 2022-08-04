"""Модуль, содержащий модели частотных сканов, полученных на DLS-82E.

"""

import numpy as np
import pandas as pd
import tensorflow as tf


class FrequencyScan(tf.Module):
    """Модель моноэкспоненциального частотного скана с показателем p.

    Модель моноэкспоненциального частотного скана с учётом показателя
    нелинейности-неэкспоненциальности p. Модель позволяет вычислять 
    частотный скан по заданным параметрам, а также идентифицировать 
    параметры модели частотного скана по экспериментальным данным методом 
    простого градиентного спуска.

    Термин "моноэкспоненциальный" подразумевает, что модель имеет только
    один набор параметров сигнала релаксации (амплитуда и частота).

    """


    @property
    def amplitude(self):
        """Свойство, возвращающее текущее или принимающее новое значение
        амлитуды сгнала релаксации ёмкости в условных единицах.

        Parameters
        ----------
        val : float
            Значение амплитуды сигнала релаксации ёмкости. Может быть
            целым числом (int), числом с плавающей точкой (float),
            tf.Variable, tf.Tensor или объектом Python, преобразуемым в 
            tf.Tensor. Передаваемый объект должен иметь размерность 
            (1,).

        Returns
        -------
        numpy.float64 или tf.Varivable(dtype='float64')
            Значение амплитуды сигнала релаксации ёмкости. Если свойство
            tf_in_out == False (значение по умолчанию), то возвращаемое
            значение имеет тип numpy.float64. Если tf_in_out == True, 
            то возвращаемое значение имеет тип 
            tf.Varivable(dtype='float64').

        """
        if self._tf_in_out:
            return self._amplitude
        else:
            return self._amplitude.numpy()

    @amplitude.setter
    def amplitude(self, val):
        self._amplitude = tf.Variable(val, dtype='float64')


    @property
    def time_constant_power(self):
        """Свойство, возвращающее текущее или принимающее новое значение
        постоянной времени сгнала релаксации ёмкости в секундах.

        Parameters
        ----------
        val : float
            Значение постоянной времени сигнала релаксации ёмкости. 
            Может быть целым числом (int), числом с плавающей точкой 
            (float), tf.Variable, tf.Tensor или объектом Python,
            преобразуемым в tf.Tensor. Передаваемый объект должен иметь 
            размерность (1,).

        Returns
        -------
        numpy.float64 или tf.Varivable(dtype='float64')
            Значение постоянной времени сигнала релаксации ёмкости. Если
            свойство tf_in_out == False (значение по умолчанию), то 
            возвращаемое значение имеет тип numpy.float64. Если 
            tf_in_out == True, то возвращаемое значение имеет тип 
            tf.Varivable(dtype='float64').

        """
        if self._tf_in_out:
            return self._time_constant_power
        else:
            return self._time_constant_power.numpy()

    @time_constant_power.setter
    def time_constant_power(self, val):
        self._time_constant_power = tf.Variable(val, dtype='float64')


    @property
    def filling_pulse(self):
        """Свойство, возвращающее текущее или принимающее новое значение
        длительности импульса заполнения в секундах.

        Parameters
        ----------
        val : float
            Значение длительности импульса заполнения. Может быть целым 
            числом (int), числом с плавающей точкой (float),
            tf.Variable, tf.Tensor или объектои Python, преобразуемым в 
            tf.Tensor. Передаваемый объект должен иметь размерность 
            (1,).

        Returns
        -------
        numpy.float64 или tf.Varivable(dtype='float64')
            Значение длительности импульса заполнения. Если свойство 
            tf_in_out == False (значение по умолчанию), то возвращаемое
            значение имеет тип numpy.float64. Если tf_in_out == True, 
            то возвращаемое значение имеет тип tf.Varivable(dtype='float64').

        """
        if self._tf_in_out:
            return self._filling_pulse
        else:
            return self._filling_pulse.numpy()

    @filling_pulse.setter
    def filling_pulse(self, val):
        self._filling_pulse = tf.Variable(val, dtype='float64')


    @property
    def p_coef(self):
        """Свойство, возвращающее текущее или принимающее новое значение
        коэффициента нелинейности-неэкспоненциальности p, являющегося 
        безразмерной величиной.

        Parameters
        ----------
        val : float
            Значение коэффициента нелинейности-неэкспоненциальности.
            Может быть целым числом (int), числом с плавающей точкой 
            (float), tf.Variable, tf.Tensor или объектом Python, 
            преобразуемым в tf.Tensor. Передаваемый объект должен иметь
            размерность (1,).

        Returns
        -------
        numpy.float64 или tf.Varivable(dtype='float64')
            Значение коэффициента нелинейности-неэкспоненциальности. 
            Если свойство tf_in_out == False (значение по умолчанию), то
            возвращаемое значение имеет тип numpy.float64. Если 
            tf_in_out == True, то возвращаемое значение имеет тип 
            tf.Varivable(dtype='float64').

        """
        if self._tf_in_out:
            return self._p_coef
        else:
            return self._p_coef.numpy()

    @p_coef.setter
    def p_coef(self, val):
        self._p_coef = tf.Variable(val, dtype='float64')


    @property
    def fit_p_coef(self):
        """bool: Если True - выполняется идентификация коэффициента p."""
        return self._fit_p_coef

    @fit_p_coef.setter
    def fit_p_coef(self, val):
        self._fit_p_coef = val


    @property
    def learning_rate(self):
        """float: Скорость градиентного спуска."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, val):
        self._learning_rate = val
    

    @property
    def n_iters(self):
        """int: максимальное количество итераций при идентификации модели."""
        return self._n_iters

    @n_iters.setter
    def n_iters(self, val):
        self._n_iters = val


    @property
    def stop_val(self):
        """float: Минимальное значение разницы в среднеквадратической ошибке.
        
        Если stop_val не None (значение по умолчанию), то идентификация
        останавливается, когда модуль разницы между среднеквадратической
        ошибкой на предыдущей и текущей итерациях меньше stop_val.

        """
        return self._stop_val

    @stop_val.setter
    def stop_val(self, val):
        self._stop_val = val


    @property
    def verbose(self):
        """bool: Вывод дополнительной информации.

        Если True - выводится дополнительная информация при 
        идентификации параметров модели.

        Дополнительная информация выводится в консоль и имеет следующий 
        вид:
        iter # номер итерации
        amp: значение амплитуды процесса релаксации
        tau: значение постоянной времени процесса релаксации
        p: значение коэффициента p
        Loss: значение среднеквадратической ошибки

        """
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        self._verbose = val

        
    @property
    def tf_in_out(self):
        """bool: Флаг, определяющий тип выводимых параметров модели.

        Если True, то значения, возвращаемые свойствами amplitude, 
        time_constant_power, filling_pulse, p_coef имеют тип 
        tf.Varivable(dtype='float64'), иначе numpy.float64.

        """
        return self._tf_in_out

    @tf_in_out.setter
    def tf_in_out(self, val):
        self._tf_in_out = val
    

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
        """Инициализация модели моноэкспоненциального частотного скана.

        Parameters
        ----------
        amplitude : float, default=3.5
            Значение амплитуды сигнала релаксации ёмкости в условных
            единицах. Является начальным значением амплитуды при
            идентификации параметров модели.
        time_constant_power : float, default=-2.0
            Значение постоянной времени сигнала релаксации ёмкости в 
            секундах. Является начальным значением постоянной времени
            сигнала релаксации ёмкости при идентификации параметров
            модели.
        filling_pulse : float, default=20*10**-6
            Значение длительности импульса заполнения в секундах. Не 
            изменяется во время идентификации.
        p_coef : float, default=1.0
            Значение коэффициента нелинейности-неэкспоненциальности.
            Безразмерная величина. Является начальным значением при
            коэффициента нелинейности-неэкспоненциальности при 
            идентификации параметров модели.
        fit_p_coef : bool, default=True
            Если fit_p_coef == True, при идентификации параметров модели
            происходит уточнение коэффициента нелинейности-неэкспоненциальности
            p, если fit_p_coef == False, то коэффициент p остаётся
            неизменным.
        learning_rate : float, default=0.1
            Скорость градиентного спуска.
        n_iters : int, default=1000
            Максимальное количество итераций при идентификации 
            параметров модели.
        stop_val : float, default=None
            Минимальное изменение среднеквадратической ошибки при
            идентификации. Данное значение необходимо для ранней остановки
            алгоритма идентификации параметров. Если stop_val является 
            None, то выполняется заданное максимальное количество итераций.
            Если задано другое значение, то идентификация останавливается,
            когда значение модуля разности текущей и пердшествующей
            среднеквадратической ошибки становится меньше stop_val. В
            псевдокоде условие остановки идентификации можно записать
            следующим образом:
            abs(previous_mse - current_mse) < stop_val.
        verbose : bool, default=False
            Если verbose == True, то при идентификации в консоль выводится
            дополнительная информация, имеющая следущий вид:
                iter # номер итерации
                amp: значение амплитуды процесса релаксации
                tau: значение постоянной времени процесса релаксации
                p: значение коэффициента p
                Loss: значение среднеквадратической ошибки
        tf_in_out : bool, default=False
            Флаг, определяющий тип выводимых параметров модели.
            Если True, то значения, возвращаемые свойствами amplitude, 
            time_constant_power, filling_pulse, p_coef имеют тип 
            tf.Varivable(dtype='float64'), иначе numpy.float64.
        **kwargs
            Дополнительные аргументы, определяемые классом родителем 
            tf.Module.

        """

        super().__init__(**kwargs)

        self.tf_in_out = tf_in_out

        self.amplitude = amplitude
        self.time_constant_power = time_constant_power
        self.filling_pulse = filling_pulse
        self.p_coef = p_coef
        
        self.fit_p_coef = fit_p_coef
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.stop_val = stop_val
        self.verbose = verbose


    def _get_phi(self,
                 frequency_powers
                ):
        """Метод, вычисляющий функцию phi без учёта масштабного 
        коэффициента M [1]_.

        Parameters
        ----------
        frequency_powers : tf.Variable(dtype='float64')
            Одномерный массив, содержащий значения десятичных логарифмов
            частоты точек на частотном скане. Логарифм берётся от частоты в Гц.

        Returns
        -------
        tf.Variable(dtype='float64')
            Одномерный массив, содержащий значения функции phi для 
            каждого значения frequency_powers.

        References
        ----------
        .. [1] Krylov V. P., Bogachev A. M., Pronin T. Yu. Deep level 
        relaxation spectroscopy and non-destructive testing of potential
        defects in the semiconductor electronic component base. 
        Radiopromyshlennost, 2019, vol. 29, no. 2, pp. 35–44 (In 
        Russian). DOI: 10.21778/2413-9599-2019-29-2-35-44.

        """
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
        """Метод, вычисляющий масштабный коэффициент М [1]_.

        Коэффициент М = 1/max(phi) при текущих значениях импульса 
        постоянной времени, импульса заполнения.
        Максимум определяется методом градиентного спуска.

        Parameters
        ----------
        learning_rate : float, default=0.1
            Скорость градиентного спуска при поиске максимума phi.
        n_iters : int, default=100
            Максимальное количество итераций при поиску максимума
        stop_val : float, default=None
            Минимальное изменение значения максимума. Данный параметр 
            необходим для ранней остановкиалгоритма поиска максимума. 
            Если stop_val является None, то выполняется заданное 
            максимальное количество итераций. Если задано другое 
            значение, то поиск максимума останавливается, когда модуль
            разницы между текущим и предшествующим значением максимума
            становится меньше stop_val. В псевдокоде данное условие 
            можно записать следующим образом:
            abs(previous_max - current_max) < stop_val.


        References
        ----------
        .. [1] Krylov V. P., Bogachev A. M., Pronin T. Yu. Deep level 
        relaxation spectroscopy and non-destructive testing of potential
        defects in the semiconductor electronic component base. 
        Radiopromyshlennost, 2019, vol. 29, no. 2, pp. 35–44 (In 
        Russian). DOI: 10.21778/2413-9599-2019-29-2-35-44.

        """
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
        
        
    def __call__(self, f_powers):
        """Значение сигнала DLTS.

        Метод вычисляет сигнал DLTS - сигнал на выходе коррелятора 
        спектрометра DLS-82E для каждого значения в массиве f_powers [1]_.

        Parameters
        ----------
        f_powers : array_like
            Одномерный массив, содержащий значения десятичных логарифмов
            частоты  точек на частотном скане. Данный параметр также может 
            быть целым числом (int) или числом с плавающей точкой (float), а
            также tf.Tensor или любым объектом Python, который может быть
            преобразован в tf.Tensor. Логарифм берётся от частоты в Гц.

        Returns
        -------
        tf.Tensor 
            сигнал на выходе коррелятора спектрометра DLS-82E для каждого
            значения в массиве f_powers.

        References
        ----------
        .. [1] Krylov V. P., Bogachev A. M., Pronin T. Yu. Deep level 
        relaxation spectroscopy and non-destructive testing of potential
        defects in the semiconductor electronic component base. 
        Radiopromyshlennost, 2019, vol. 29, no. 2, pp. 35–44 (In 
        Russian). DOI: 10.21778/2413-9599-2019-29-2-35-44.

        """
        frequency_powers = tf.Variable(f_powers, dtype='float64')
        
        M = self._get_M(learning_rate=0.2,
                        n_iters=100,
                        stop_val = 10**-10)
        
        phi = self._get_phi(frequency_powers)
        
        return self._amplitude * tf.pow(M * phi, self._p_coef)
    
    
    def fit(self,
            f_powers,
            dlts_vals,
           ):
        """Идентификация параметров модели.

        Метод находит оптимальные параметры модели при помощи градиентного
        спуска. 
        Градиент вычисляется при помощи библиотеки TensorFlow. 
        Функция ошибки - среднеквадратическое отклонение между 
        экспериментальными данными и моделью, соответственно алгоритм ищет 
        значения параметров модели при которых достигается минимум этой 
        функции.
        Реализована ранняя остановка алгоритма (определяется свойством 
        stop_val): алгоритм останавливается, если модуль разницы между
        значениями функции ошибки на текущей и предыдущей итерациях становится
        меньше заданного значения.
        Реализован вывод дополнительных данных в консоль для контроля процесса
        идентификации (определяется свойством verbose). Данные, выводимые в 
        консоль имеют следующий вид:
            iter # номер итерации
            amp: значение амплитуды процесса релаксации
            tau: значение постоянной времени процесса релаксации
            p: значение коэффициента p
            Loss: значение среднеквадратической ошибки

        Parameters
        ----------
        f_powers : array_like
            Одномерный массив, содержащий значения десятичных логарифмов
            частоты на частотном скане (экспериментальных данных). Логарифм
            берётся от частоты в Гц.
        dlts_vals : array_like
            Одномерный массив, содержащий значения сигнала DLTS точек на 
            частотном скане (экспериментальных данных). Значения сигнала DLTS
            имеют ту же единицу измерения (тот же масштаб), что и амплитуда
            сигнала релаксации ёмкости (одни из параметров данной модели).

        Returns
        -------
        fit_results : pd.DataFrame
            pd.DataFrame с параметрами модели и значениями функции ошибки на
            каждой итерации.

        """
        frequency_powers = tf.Variable(f_powers, dtype='float64')
        dlts = tf.Variable(dlts_vals, dtype='float64')

        
        prev_loss = tf.Variable(np.inf, dtype='float64')
        
        fit_results = pd.DataFrame(columns=['amplitude', 
                                            'time_constant_power', 
                                            'p_coef', 
                                            'loss'])
        
        for _ in range(self._n_iters):
            with tf.GradientTape() as tape:
                predicted_dlts = self.__call__(frequency_powers)
                current_loss = tf.reduce_mean(tf.square(dlts - predicted_dlts))
                
            if self._fit_p_coef:
                dampl, dtime_const_pow, dp_coef = tape.gradient(current_loss,
                    [self._amplitude, self._time_constant_power, self._p_coef])
            else:
                dampl, dtime_const_pow = tape.gradient(current_loss, 
                    [self._amplitude, self._time_constant_power])
                
            fit_results.loc[_, 'amplitude'] = self._amplitude.numpy()
            fit_results.loc[_, 'time_constant_power'] = self._time_constant_power.numpy()
            fit_results.loc[_, 'p_coef'] = self._p_coef.numpy()
            fit_results.loc[_, 'loss'] = current_loss.numpy()     
            
            if self._verbose:
                print('iter #', _)
                print('amp:',self.amplitude)
                print('tau:',self.time_constant_power)
                print('p:', self.p_coef)
                if self._tf_in_out:
                    print('Loss:', current_loss)
                else:
                    print('Loss:', current_loss.numpy())
                
            self._amplitude.assign_sub(self._learning_rate * dampl)
            self._time_constant_power.assign_sub(self._learning_rate * dtime_const_pow)
            if self._fit_p_coef:
                self._p_coef.assign_sub(self._learning_rate * dp_coef)
                
            if self._stop_val is not None:
                if tf.abs(current_loss - prev_loss) < self._stop_val:
                    break
                    
            prev_loss = current_loss
            
        return fit_results
from joblib import delayed
from joblib import Parallel
from matplotlib import pyplot as plt
import numpy as np
from os import listdir
import pandas as pd
import re

from pydlts.fsmodels import SklSingleExpFrequencyScan
from pydlts.fsplots import plot_model
from pydlts.fsplots import plot_loss_path
from pydlts.fsplots import plot_experimental_points



DATASETS_PATH = '../datasets'
MODELS_PATH = '../models'
PLOTS_PATH = '../plots'
RAW_DATA_PATH = '../raw_data'
REPORT_PATH = '../Отчёт'



class DataReaderDLS82E():
    """
    Reader for experimental data obtained on the measuring complex for DLTS. 
    It can read csv-files and experimental data split into two files (it's
    the obsolete and akward format). One of which contains the results of DLTS 
    measurements, the other contains the results of temperature measurements. 
    It puts the data to pandas.DataFrame, later it can write it to csv or hdf5 
    files. It's also posible to show a plot of the data and write this image 
    to .svg or .jpg.
    
    Attributes
    ----------
    data : pandas.DataFrame
        It's a DataFrame containing the experimental data.
        There are several columns containing all necessary information:
        'time' - date and time than the measurement was performed, 
        'frequency_hz' - filling pulse frequency in hertz, 
        'dlts_v' - value of DLTS-signal in volts, 
        'temperature_k' - temperature in kelvin,
        'dlts_pf' - value of DLTS-signal in picofarads,
        'bs' - bridge sensitivity in picofarads,
        'ls' - selector sensitivity in millivolts,
        'f_pulse' - duration of filling pulse in microseconds,
        'u1' - level of filling pulse 1 in volts,
        'ur' - reverse bias in volts,
        'time_between_meas' - time between two measurements in seconds,
        'integral_time' - time constant of the integrating circuit in seconds,
        'specimen_name' - name of the specimen.
        
    Methods
    -------
    read_from_d_t(d_file_name, t_file_name, encoding='cp1251')
        Read data from text files with experimental data.
    read_from_csv(fname, encoding=None)
        Read data from csv-file created by an instance of 
        the ExperimentalDataReader class.
    read_from_hdf(fname, key=None)
        Read data from a binary file in the HDF5 format.
    set_specimen_name(specimen_name)
        Write the specimen name to the specimen_name 
        column of the self.data attribute.
    set_bs(bs=np.nan)
        Write the bridge sensitivity value to the 
        bs column of the self.data attribute.
    set_ls(ls=np.nan)
        Write the selector sensitivity to the 
        ls column of the self.data attribute.
    set_f_pulse(f_pulse=np.nan)
        Write the duration of the filling pulse to 
        the f_pulse column of the self.data attribute.
    set_u1(u1=np.nan)
        Write the level of filling pulse 1 to 
        the u1 column of the self.data attribute.
    set_ur(ur=np.nan)
        Write the value of the reverse bias to 
        the ur column of the self.data attribute.
    set_time_between_meas(time=np.nan)
        Write the value of time between measurements to 
        the time_between_meas column of the self.data attribute.
    set_integral_time(time=np.nan)
        Write the value of the time constant of the integrating circuit to 
        the specimen_name column of the self.data attribute.
    compute_dlts_pf()
        Convert values of DLTS-signal in volts to values in picofarads and 
        write them to the dlts_pf column of the self.data attribute.
    to_csv(fname)
        Write the self.data DataFrame to the csv-file.
    to_hdf(fname, key='data')
        Write the self.data DataFrame to the binary file in the HDF5 format.
    get_plot()
        Make a plot of the experimental data.
    """
    
    def __init__(self):
        """
        Class constructor.
        Creates an instance of ExperimentalDataReader class with empty data attribute.
        """

        self.data = pd.DataFrame(columns=['time', 
                                          'frequency_hz', 
                                          'dlts_v', 
                                          'temperature_k',
                                          'dlts_pf',
                                          'bs',
                                          'ls',
                                          'f_pulse',
                                          'u1',
                                          'ur',
                                          'time_between_meas',
                                          'integral_time',
                                          'specimen_name'])
        
        
    def read_from_d_t(self, d_file_name, t_file_name, encoding='cp1251'):
        """
        Read data from text files with experimental data, merge the pieces 
        of the data into one DataFrame, and write it to the self.data attribute.
        
        Parameters
        ----------
        d_file_name : str 
            String containing the name of the file with temperature.
        t_file_name : str
            String containing the name of the file with DLTS.
        encoding : str
            Encoding of text files with experimental data. 
            Encodings for both files must be the same. 
            The default value is 'cp1251'.
        """
        
        dlts_data = pd.read_csv(d_file_name, 
                                sep=' ',
                                encoding=encoding,
                                comment='#',
                                skipinitialspace=True,
                                usecols=[0, 2, 3],
                                header=None,
                                names=['time', 'frequency_hz', 'dlts_v'])
        
        temperature_data = pd.read_csv(t_file_name,
                                       sep='[;\s]+',
                                       encoding=encoding,
                                       usecols=[0, 3],
                                       header=None,
                                       names=['time', 'temperature_k'],
                                       skiprows=1,
                                       engine='python')
        
        with open(d_file_name, 'r', encoding=encoding) as f:
            date_str = re.findall('\d\d\.\d\d\.\d\d\d\d', f.read())
        
        dlts_data['time'] = dlts_data.time + ' ' + date_str
        dlts_data['time'] = pd.to_datetime(dlts_data.time, format='%H:%M:%S %d.%m.%Y')
        
        temperature_data['time'] = temperature_data.time + ' ' + date_str
        temperature_data['time'] = pd.to_datetime(temperature_data.time, format='%H:%M:%S %d.%m.%Y')
        
        temperature_data.set_index('time', inplace=True)
        temperature_data = temperature_data.resample('S').ffill()
        
        column_list = ['time', 'frequency_hz', 'dlts_v', 'temperature_k']
        self.data[column_list] = dlts_data.merge(right=temperature_data, how='left', on='time')
    
    
    def read_from_csv(self, fname, encoding=None):
        """
        Read data from csv-file created by an instance of the ExperimentalDataReader class.
        
        If encoding=None, the pd.read_csv() is called with default encoding, else
        it is called with given encoding.
        
        Parameters
        ----------
        fname : str
            String containing the name of the csv-file.
        encoding : str
            Encoding of csv-file with experimental data. 
            The default value is None.
        """
        
        if encoding is None:
            self.data = pd.read_csv(fname)
        else:
            self.data = pd.read_csv(fname, encoding=encoding)
    
    
    def read_from_hdf(self, fname, key=None):
        """
        Read data from a binary file in the HDF5 format.
        
        If key=None, the pd.read_hdf() is called with default key, else
        it is called with given key.
        
        Parameters
        ----------
        fname : str
            String containing the name of the hdf-file.
        key : str
            Key of the dataset in the hdf-file with experimental data. 
            The default value is None.
        """
        
        if key is None:
            self.data = pd.read_hdf(fname)
        else:
            self.data = pd.read_hdf(fname, key)
    
    
    def set_specimen_name(self, specimen_name):
        """
        Write the specimen name to the specimen_name column of the self.data attribute.
        There must be only one specimen name in the all dataset.
        
        Parameters
        ----------
        specimen_name : str
            String containing the name of the specimen.
        """
        
        self.data.specimen_name = specimen_name
    
    
    def set_bs(self, bs=np.nan):
        """
        Check the bridge sensitivity value and write it to the bs column of the self.data attribute.
        
        Parameters
        ----------
        bs : float
            The bridge sensitivity value in picofarads. It's a kind of categorical value. It must be 1, 10, 100, 
            1000 or np.nan. In other cases, a ValueError is raised. The default value is np.nan.
        
        Raises
        ------
        ValueError
            If the bs contains wrong value.
        """
        
        allowed_values = [1, 10, 100, 1000, np.nan]
        
        error_message = 'bs value must be ' + ', '.join((str(_) for _ in allowed_values[:-1])) + ' or NaN'
        
        if bs in allowed_values:
            self.data.bs = bs
        else:
            raise ValueError(error_message)
    
    
    def set_ls(self, ls=np.nan):
        """
        Check the selector sensitivity value and write it to the ls column of the self.data attribute.
        
        Parameters
        ----------
        ls : float
            The selector sensitivity value in volts. It's a kind of categorical value. It must be 1, 2, 5, 10, 
            20, 50, 100, 200, 500, 1000 or np.nan. In other cases, a ValueError is raised. 
            The default value is np.nan.
        
        Raises
        ------
        ValueError
            If the ls contains wrong value.
        """
        
        allowed_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, np.nan]
        
        error_message = 'ls value must be ' + ', '.join((str(_) for _ in allowed_values[:-1])) + ' or NaN'
        
        if ls in allowed_values:
            self.data.ls = ls
        else:
            raise ValueError(error_message)
    
    
    def set_f_pulse(self, f_pulse=np.nan):
        """
        Write the duration of the filling pulse to the f_pulse column of the self.data attribute.
        
        Parameters
        ----------
        f_pulse : float
            The duration of the filling pulse in microseconds. The default value is np.nan.
        """
        
        self.data.f_pulse = f_pulse
    
    
    def set_u1(self, u1=np.nan):
        """
        Write the level of filling pulse 1 to the u1 column of the self.data attribute.
        
        Parameters
        ----------
        u1 : float
            The level of filling pulse 1 in volts. The default value is np.nan.
        """
        
        self.data.u1 = u1
    
    
    def set_ur(self, ur=np.nan):
        """
        Write the value of the reverse bias to the ur column of the self.data attribute.
        
        Parameters
        ----------
        ur : float
            The value of the reverse bias in volts. The default value is np.nan.
        """
        
        self.data.ur = ur
        
        
    def set_time_between_meas(self, time=np.nan):
        """
        Write the value of time between measurements to 
        the time_between_meas column of the self.data attribute.
        
        Parameters
        ----------
        time : float
            The value of time between measurements in seconds. The default value is np.nan.
        """
        
        self.data.time_between_meas = time
    
    
    def set_integral_time(self, time=np.nan):
        """
        Write the value of the time constant of the integrating circuit 
        to the specimen_name column of the self.data attribute.
        
        Parameters
        ----------
        time : float
            The value of the time constant of the integrating circuit in seconds.
            It's a kind of categorical value. It must be 0.3, 1, 3, 10, 30 or np.nan. 
            In other cases, a ValueError is raised. The default value is np.nan.
            
        Raises
        ------
        ValueError
            If the time contains wrong value.
        """
        
        allowed_values = [0.3, 1, 3, 10, 30, np.nan]
        
        error_mesage = 'time value must be ' + ', '.join((str(_) for _ in allowed_values[:-1])) + ' or NaN'
        
        if time in allowed_values:
            self.data.integral_time = time
        else:
            raise ValueError(error_mesage)
    
    def compute_dlts_pf(self):
        """
        Convert values of DLTS-signal in volts to values in picofarads and 
        write them to the dlts_pf column of the self.data attribute.
        """
        
        self.data.dlts_pf = self.data.dlts_v / ((10 / self.data.bs)*(10000 / self.data.ls))

    
    def to_csv(self, fname):
        """
        Write the self.data DataFrame to the csv-file.
        
        Parameters
        ----------
        fname : string
            The name of the csv-file.
        """
        
        self.data.to_csv(fname, index=False)
    
    
    def to_hdf(self, fname, key='data'):
        """
        Write the self.data DataFrame to the binary file in the HDF5 format.
        
        Parameters
        ----------
        fname : string
            The name of the hdf-file.
        key : string
            The key of the dataset in the hdf-file. The default value is 'data'.
        """
        
        self.data.to_hdf(fname, key)
        
    
    def get_plot(self):
        """
        Make a plot of the experimental data.
        
        Returns
        -------
        fig : `~.figure.Figure`
        ax : `.axes.Axes` or array of Axes
        """
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [4, 1]})
        
        self.data.plot(ax=ax[0],
                       x='frequency_hz',
                       y='dlts_pf',
                       kind='scatter',
                       logx=True,
                       xlabel='Frequency, Hz',
                       ylabel='DLTS, pF',
                       grid=True,
                      )
        
        self.data.plot(ax=ax[1],
                       x='frequency_hz',
                       y='temperature_k',
                       kind='scatter',
                       logx=True,
                       xlabel='Frequency, Hz',
                       ylabel='Temperature, K',
                       grid=True,
                       ylim=(np.round(self.data.temperature_k.min() - 1, 0), np.round(self.data.temperature_k.max() + 1, 0))
                      )
        
        return fig, ax



class BatchDataReaderDLS82E():


    def __init__(self, 
                 integral_time = 3.0,
                 time_between_meas = 3.5,
                 raw_data_path=RAW_DATA_PATH, 
                 datasets_path=DATASETS_PATH, 
                 plots_path=PLOTS_PATH):
        
        self.integral_time = integral_time
        self.time_between_meas = time_between_meas

        self.raw_data_folder = raw_data_path
        self.datasets_folder = datasets_path
        self.plots_folder = plots_path


    def _get_file_names(self):
        
        condition = lambda x: not x.startswith('Температура')
        file_name_list = filter(condition, listdir(self.raw_data_folder))
        
        files = pd.DataFrame({'data_files': file_name_list})
        files['temperature_files'] = 'Температура_' + files + '.txt'

        return files


    def _get_params(self, files):

        files['patterns'] = files.data_files.str.split('_')
        
        get_spc_name = lambda x: ' '.join(x[:2])
        files['specimen_name'] = files.patterns.apply(get_spc_name)
        
        get_bs = lambda x: int(x[3].strip('пФ'))
        files['bs'] = files.patterns.apply(get_bs)
        
        get_u1 = lambda x: float(re.findall(string=x[5], pattern=r'[+-]?\d+\.?\d*')[0])
        files['u1'] = files.patterns.apply(get_u1)
        
        get_ur = lambda x: float(re.findall(string=x[5], pattern=r'[+-]?\d+\.?\d*')[-1])
        files['ur'] = files.patterns.apply(get_ur)
        
        get_ls = lambda x: int(x[6].strip('мВ'))
        files['ls'] = files.patterns.apply(get_ls)
        
        get_f_pulse = lambda x: float(x[7].strip('мкс'))
        files['f_pulse'] = files.patterns.apply(get_f_pulse)
        
        files['time_between_meas'] = self.time_between_meas
        files['integral_time'] = self.integral_time
        
        return files.drop('patterns', axis='columns')


    def _get_meta_data(self):

        files = self._get_file_names()

        return self._get_params(files)


    def _init_data_reader(self, meta_data):
        data_reader = DataReaderDLS82E()

        d_file_name = self.raw_data_folder + '/' + meta_data.data_files
        t_file_name = self.raw_data_folder + '/' + meta_data.temperature_files

        data_reader.read_from_d_t(d_file_name=d_file_name, t_file_name=t_file_name)

        data_reader.set_bs(meta_data.bs)

        data_reader.set_ls(meta_data.ls)

        data_reader.set_f_pulse(meta_data.f_pulse)

        data_reader.compute_dlts_pf()

        data_reader.set_specimen_name(meta_data.specimen_name)

        data_reader.set_u1(meta_data.u1)

        data_reader.set_ur(meta_data.ur)

        data_reader.set_time_between_meas(meta_data.time_between_meas)

        data_reader.set_integral_time(meta_data.integral_time)

        return data_reader


    def read_data(self):
        meta_data = self._get_meta_data()

        for i, meta in meta_data.iterrows():
            try:
                data_reader = self._init_data_reader(meta_data=meta)
                
            except FileNotFoundError:
                print(f'№{i:<3}\t{meta_data.data_files[i]:<70}\t- FileNotFoundError')
                
            else:
                data_reader.to_csv(self.datasets_folder + '/' + meta_data.data_files[i] + '.csv')

                data_reader.get_plot()
                plt.savefig(self.plots_folder + '/' + meta_data.data_files[i] + '.pdf')

                plt.close('all')
                print(f'№{i:<3}\t{meta_data.data_files[i]:<70}\t- Ok')



class BatchSingleExp():


    def __init__(self,
                 fit_p_coef = True,
                 learning_rate = 0.05,
                 n_iters = 1000,
                 stop_val = 10**-10,
                 verbose = False,
                 datasets_path=DATASETS_PATH,
                 plots_path=PLOTS_PATH,
                 models_path=MODELS_PATH,
                 n_jobs=1,
                ):

        self.fit_p_coef = fit_p_coef
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.stop_val = stop_val
        self.verbose = verbose

        self.datasets_folder = datasets_path
        self.plots_folder = plots_path
        self.models_folder = models_path
        self.n_jobs = n_jobs


    def _get_file_names(self):
        return [self.datasets_folder + '/' + name for name in listdir(self.datasets_folder)]


    def _read_datasets(self, file_names):

        read_dataset = lambda name: pd.read_csv(name,
                                                header=0,
                                                parse_dates=[0],
                                                infer_datetime_format=True
                                               )

        return [[f_name, read_dataset(f_name)] for f_name in file_names]


    def _fit_model(self, df):

        f_pulse = df.f_pulse[0] * 10 ** -6

        X_train = df.frequency_hz.to_numpy()
        X_train = np.log10(X_train)

        y_train = df.dlts_pf.to_numpy()

        model = SklSingleExpFrequencyScan(filling_pulse=f_pulse,
                                          fit_p_coef=self.fit_p_coef,
                                          learning_rate=self.learning_rate,
                                          n_iters=self.n_iters,
                                          stop_val=self.stop_val,
                                          verbose=self.verbose
                                         )

        model.fit(X=X_train, y=y_train)

        return model.fit_results_


    def _get_text_params(self,fit_result):
        time_constant_power = fit_result.time_constant_pow_0
        f_pulse = fit_result.filling_pulse
        p = fit_result.p_coef
        amp = fit_result.amplitude_0
        mse = fit_result.loss

        text = '\n'.join(['$\\log_{10}(\\tau)$ = ' + f'{time_constant_power:.4f} ' + '$\\log_{10}$(с)',
                          f'$\\tau$ = {10**time_constant_power:.4e} с',
                          f'$A$ = {amp:.4e} пФ',
                          f'$p$ = {p:.4f}',
                          f'MSE = {mse:.4e} $пФ^2$',
                          f'RMSE = {np.sqrt(mse):.4e} пФ'
                         ])

        return text


    def _get_additional_text(self, df, fit_results_):

        frequency_powers = np.log10(df.frequency_hz.to_numpy())
        dlts_values = df.dlts_pf.to_numpy()
        f_pulse = df.f_pulse[0] * 10 ** (-6)
        
        text_1 = '\n'.join([f'Образец: {df.specimen_name[0]}',
                            f'$T$ = {df.temperature_k.mean():.1f} К',
                            f'$U_1$={df.u1[0]} В',
                            f'$U_R$={df.ur[0]} В',
                            f'$t_1$ = {f_pulse:.4e} с'
                           ])
        
        text_2 = '\n'.join(['Конечные значения:', self._get_text_params(fit_results_.iloc[-1, :])])
        
        return text_1, text_2


    def _print_results(self, df, fit_results_):

        frequency = df.frequency_hz.to_numpy()
        frequency_powers = np.log10(frequency)
        actual_dlts = df.dlts_pf.to_numpy()

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5))

        ax0 = plot_model(X = frequency_powers,
                         y = actual_dlts,
                         model_class = SklSingleExpFrequencyScan,
                         fit_results_ = fit_results_,
                         plot_exps = False,
                         ax=ax0
                        )
        ax0.set_ylabel('Сигнал DLTS, пФ')

        ax1 = plot_loss_path(fit_results_, ax=ax1)

        title = f'{df.specimen_name[0]} T={df.temperature_k.mean():.1f} K, $U_1$={df.u1[0]} V, $U_R$={df.ur[0]} V'
        plt.suptitle(title, y=1.01)

        text = '\n\n'.join(self._get_additional_text(df, fit_results_))

        x = 0.4 * (max(ax1.get_xlim()) - min(ax1.get_xlim())) + min(ax1.get_xlim())
        y = 0.95 * max(ax1.get_ylim())
        fontsize=10
        bbox_dict = {'facecolor':'white', 'alpha':0.8, 'edgecolor':'gray'}
            
        ax1.text(x, y, text, fontsize=fontsize, verticalalignment='top', bbox=bbox_dict)
        
        return fig, (ax0, ax1)


    def _write_model(self, f_name, df):

        fit_results_ = self._fit_model(df)

        fig, ax = self._print_results(df, fit_results_)

        last_row = fit_results_.iloc[-1]
        final_model = SklSingleExpFrequencyScan(filling_pulse=last_row.filling_pulse)
        final_model.exps_params_ = [ [ last_row.time_constant_pow_0, last_row.amplitude_0 ] ]
        final_model.p_coef_ = last_row.p_coef

        model_df = df.copy()

        frequency_powers = np.log10(df.frequency_hz.to_numpy())
        model_df['dlts_pf_model'] = final_model.predict(frequency_powers)
        model_df['p_coef_model'] = final_model.p_coef_

        if model_df['p_coef_model'].isna().any():
            message = f_name.split('/')[-1].rstrip('.csv') + '_model - ERROR'
        else:
            message = f_name.split('/')[-1].rstrip('.csv') + '_model - OK'

        model_df['time_constant_power_model'] = final_model.exps_params_[0, 0]
        model_df['time_constant_model'] = 10 ** final_model.exps_params_[0, 0]
        model_df['amplitude_model'] = final_model.exps_params_[0, 1]

        mse = np.square(model_df.dlts_pf - model_df.dlts_pf_model)
        model_df['rmse_model'] = np.sqrt(mse)

        file_name = self.models_folder + '/' + f_name.split('/')[-1].rstrip('.csv') + '_model' + '.csv'
        model_df.to_csv(file_name, index=False)
        
        file_name = self.plots_folder + '/' + f_name.split('/')[-1].rstrip('.csv') + '_model' + '.pdf'
        plt.savefig(file_name, bbox_inches='tight')
        
        plt.close('all')
        
        return message




    def create_models(self):
        f_names = self._get_file_names()
        df_list = self._read_datasets(f_names)

        messages = Parallel(n_jobs=self.n_jobs)(delayed(self._write_model)(f_name, df) for f_name, df in df_list)

        return messages



class AutoReport():


    def __init__(self,
                 plots_path=PLOTS_PATH,
                 models_path=MODELS_PATH,
                 report_path=REPORT_PATH,
                ):

        self.plots_folder = plots_path
        self.models_folder = models_path
        self.report_folder = report_path


    def _get_file_names(self):
        return [self.models_folder + '/' + name for name in listdir(self.models_folder)]


    def _read_datasets(self, file_names):

        read_dataset = lambda name: pd.read_csv(name,
                                                header=0,
                                                parse_dates=[0],
                                                infer_datetime_format=True
                                               )

        datasets = [[f_name, read_dataset(f_name)] for f_name in file_names]

        get_timestamp = lambda x: x[1].time[0]

        datasets.sort(key=get_timestamp)

        return datasets


    def _create_section(self, name):
        template = r'\section{{{name}}}'
        return template.format(name=name) + '\n'


    def _create_subsection(self, name):
        template = r'\subsection{{{name}}}'
        return template.format(name=name) + '\n'


    def _create_info_about_exp(self, dataset, caption, label):

        template = r'''\begin{{table}}[!ht]
    \centering
    \caption{{{caption}}}
    \begin{{tabular}}{{|l|l|}}
        \hline
        Параметр                                       & Значение                  \\ \hline
        Образец                                        & {specimen_name:<25s} \\ \hline
        Балансировка моста                             & см. рабочий журнал        \\ \hline
        Чувствительность моста, пФ                     & {bs:<25d} \\ \hline
        Чувствительность селектора, мВ                 & {ls:<25d} \\ \hline
        $U_R$, В                                       & {ur:<25.2f} \\ \hline
        $U_1$, В                                       & {u1:<25.2f} \\ \hline
        Температура в лаборатории, $^\circ C$          & см. рабочий журнал        \\ \hline
        Температура образца в начале сканирования, $K$ & {temperature:<25.2f} \\ \hline
        Температура, установленная на КТХ, $^\circ C$  & см. рабочий журнал        \\ \hline
        Время начала сканирования                      & {time:<25s} \\ \hline
        Дата                                           & {date:<25s} \\ \hline
    \end{{tabular}}
    \label{{{label}}}
\end{{table}}
'''
    
        first_row = dataset.loc[0, :]
        text = template.format(caption = caption,
                               label = label,
                               specimen_name = first_row.specimen_name,
                               bs = first_row.bs,
                               ls = first_row.ls,
                               ur = first_row.ur,
                               u1 = first_row.u1,
                               temperature = first_row.temperature_k,
                               time = str(first_row.time.time()),
                               date = str(first_row.time.date())
                              )
        
        return text + '\n'


    def _create_info_about_model(self, dataset, caption, label):
        template = r'''\begin{{table}}[!ht]
    \centering
    \caption{{{caption}}}
    \begin{{tabular}}{{|l|r|}}
        \hline
        Параметр                                       & Значение                  \\ \hline
        $\log_{{10}}(\tau)$, $\log_{{10}}$(с)              & {tau_pow:<25e} \\ \hline
        $\tau$, с                                      & {tau:<25e} \\ \hline
        $A$, пФ                                        & {amplitude:<25e} \\ \hline
        $p$                                            & {p_coef:<25e} \\ \hline
        MSE, пФ$^2$                                    & {mse:<25e} \\ \hline
        RMSE пФ                                        & {rmse:<25e} \\ \hline
    \end{{tabular}}
    \label{{{label}}}
\end{{table}}
'''
    
        first_row = dataset.loc[0, :]
        text = template.format(caption = caption,
                               label = label,
                               tau_pow = first_row.time_constant_power_model,
                               tau = first_row.time_constant_model,
                               amplitude = first_row.amplitude_model,
                               p_coef = first_row.p_coef_model,
                               mse = first_row.rmse_model ** 2,
                               rmse = first_row.rmse_model
                              )
        
        return text + '\n'


    def _create_image(self, pic_name, caption, label, size=1):
        template = r'''\begin{{figure}}[!ht]
    \centering
    \includegraphics[width={size}\textwidth]{{{pic_name}}}
    \caption{{{caption}}}
    \label{{{label}}}
\end{{figure}}
'''
        text = template.format(pic_name = pic_name,
                               caption = caption,
                               label = label,
                               size = size
                              )
        return text + '\n'


    def create_auto_report(self):

        file_names = self._get_file_names()
        datasets = self._read_datasets(file_names)

        get_model_plots_name = lambda fname, plot_path: plot_path + '/' + fname.split('/')[-1].rstrip('.csv') + '.pdf'
        model_plots_names = [get_model_plots_name(name, self.plots_folder) for (name, _) in datasets]

        get_exp_plots_name = lambda fname, plot_path: plot_path + '/' + fname.split('/')[-1].rstrip('_model.csv') + '.pdf'
        exp_plots_names = [get_exp_plots_name(name, self.plots_folder) for (name, _) in datasets]

        prep_fname = lambda fname: fname.replace('_', r'\_')


        with open(self.report_folder + '/results.tex', 'w', encoding='UTF-8') as file:
            file.write(r'%Файл сгенерирован автоматически на основе собранных и обработанных данных' + '\n\n\n')
            file.write(self._create_section('Результаты измерений'))
            file.write('В данном разделе приведены результаты измерений и идентификации параметров полученных сканов.\n')
            
            for i, (fname, data) in enumerate(datasets):
                file.write(self._create_subsection(f'Частотный скан №{i+1}'))
                file.write(self._create_info_about_exp(data, f'Параметры частотного скана №{i+1}.', f'table:frequency_scan_{i+1}'))
                
                file.write(self._create_info_about_model(data, 
                                                   f'Параметры модели частотного скана №{i+1}.',
                                                   f'table:frequency_scan_model_{i+1}'))
                
                file.write(r'\textbf{Имена файлов}' + '\n\n')
                file.write(r'Результаты измерений и параметры модели:' + '\n\n')
                file.write(r'\scriptsize' + prep_fname(fname) + '\n' + r'\normalsize' + '\n\n')
                file.write(r'График с результатами измерений:' + '\n\n')
                file.write(r'\scriptsize' + prep_fname(exp_plots_names[i]) + '\n' + r'\normalsize' + '\n\n')
                file.write(r'График с результатами идентификации модели:' + '\n\n')
                file.write(r'\scriptsize' + prep_fname(model_plots_names[i]) + '\n' + r'\normalsize' + '\n\n')
                
                file.write(self._create_image(pic_name=exp_plots_names[i], 
                                        caption=f'Частотный скан №{i+1}.', 
                                        label=f'pic:frequency_scan_{i+1}'))
                
                file.write(self._create_image(pic_name=model_plots_names[i], 
                                        caption=f'Результаты идентификации параметров модели частотного скана~№{i+1}.', 
                                        label=f'pic:frequency_scan_model{i+1}'))
                
                file.write(r'\pagebreak'+'\n\n\n')
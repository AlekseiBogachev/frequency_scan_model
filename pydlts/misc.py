import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re

class DataReader():
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
        
        with open(d_file_name, 'r') as f:
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
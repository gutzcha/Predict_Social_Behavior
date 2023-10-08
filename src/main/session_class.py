from typing import List, Union, Tuple, Any
from utils.get_project_root import get_project_root
from utils.read_matlab_file import read_matlab
import os
import os.path as osp
import pandas as pd
import numpy as np
import soundfile as sf
from utils.consts import area_name_to_number_map, area_number_to_name_map, FILES_MAPPING_PATH
from utils.helper_functions import downsample_signal, remove_mains_hum, compute_time_coherence, \
    create_binary_list_with_ordered_data, convert_time_stamps_to_frame_numbers, convert_to_numpy_array,\
    normalize_pose, pose_df_to_matrix, fix_df_col_names


from scipy.signal import coherence
# debug
import matplotlib.pyplot as plt


class SocialExpSessionBase:
    def __init__(self, data_path_dict=None):
        """
        Base class for all social experiment session data
        :param data_path_dict: a dictionary with modalities as keys and file paths as values
        """
        self.data_path_dict = data_path_dict
        self.trigger_time = None
        self.insertion_time = None

    def load_files(self, prop, is_audio=False) -> List[Union[dict, None]]:
        """
        Loads file dictionary, either a MATLAB file of WAV audio file
        :param prop: name of modality to load ('audio', 'lfp', 'multispikes', 'sniff', 'timestamps', 'video')
        :param is_audio: a boolean flag indicating if the modality is a wav audio file
        :return: list of dictionaries
        """
        file_path = self.data_path_dict.get(prop)
        if file_path is None or pd.isna(file_path):
            return [None]
        elif file_path.endswith('mat'):
            return read_matlab(file_path)
        elif file_path.lower().endswith('wav'):
            audio_data, sample_rate = sf.read(file_path)
            return [{'audio_data': audio_data, 'sample_rate': sample_rate}]
        elif 'audio excel files' in file_path.lower():
            return [{'audio_usv_data': pd.read_excel(file_path)}]
        elif "video_pose_estimation" in file_path.lower():
            return [{'pose_data': pd.read_csv(file_path)}]
        else:
            raise TypeError('Unknown data type')

    def set_trigger_time(self, trigger_time: float):
        """
        Sets trigger time based on input or "trigger_time" values in timestamps dictionary
        :param trigger_time:
        :return:
        """
        self.trigger_time = trigger_time

    def set_stimulus_insertion_time(self, insertion_frame, frame_rate):
        self.insertion_time = insertion_frame / frame_rate


class ModalityDataContainer(SocialExpSessionBase):
    def __init__(self,
                 data_dict,
                 timestamps_dict: dict,
                 sampling_rate: Union[float, int, None] = None,
                 trigger_time: Union[float, None] = None,
                 ):
        """
        base class for all time series modalities
        :param data_dict:
        :param timestamps_dict:
        :param sampling_rate:
        :param trigger_time:
        """
        super().__init__()

        self._pad_func = None  # How to crop and pad the arrays when extracting data
        self._crop_func = None
        self.timestamps = timestamps_dict

        self.data_dict = data_dict

        self.sampling_rate = sampling_rate

        self.trigger_time = 0 if trigger_time is None else trigger_time

        self.data = self.get_raw_data()

        self.time_dim = -1 # last dim

    def get_time_from_ind(self, ind):
        return ind / self.sampling_rate + self.trigger_time

    def get_max_range(self):  # TODO Check this later
        return self.timestamps['FramesTimesInSec'][-1]

    def get_adjusted_max_range(self):
        return self.timestamps['FramesTimesInSec'][-1] - self.trigger_time

    def get_max_range_inds(self):
        return int(self.sampling_rate * self.get_adjusted_max_range())

    def get_adjusted_max_range_inds(self):
        return int(self.get_adjusted_max_range()*self.sampling_rate)

    def get_raw_data(self):
        raise NotImplemented

    def _pad_array(self, array, pad_size, pad_func):
        def default_pad_func(x, n):
            return np.pad(x, ((0, 0), (n, 0)), mode='reflect')

        if pad_func is None:
            pad_func = default_pad_func
        return pad_func(array, pad_size)

    def _crop_array(self, array, n_start, n_end=None, crop_func=None):
        def default_crop_func(x, n_start, n_end):
            if n_end is None or n_end > array.shape[-1]:
                n_end = array.shape[self.time_dim]

            return x[:, int(np.round(n_start)):int(np.round(n_end))]

        if crop_func is None:
            crop_func = default_crop_func
        return crop_func(x=array, n_start=n_start, n_end=n_end)

    def get_max_freq_range(self):
        '''
        get nyquist freq
        :return:
        '''
        return [0, self.sampling_rate / 2]

    def get_data_in_range(self, array,
                          time_range: Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
                          sampling_rate=None, crop_func=None,
                          pad_func=None):
        if crop_func is None:
            crop_func = self._crop_func

        if pad_func is None:
            pad_func = self._pad_func

        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        if isinstance(time_range, tuple):
            assert len(
                time_range) == 2, f'When the time_range argument is a tuple, it must have len=2 but it was len={len(time_range)}'
            start_time, end_time = time_range
        elif isinstance(time_range, (int, float)):
            start_time = time_range
            end_time = array.shape[-1]
        else:
            raise ValueError(f'time_range must be a tuple of float number of a single float number')
        assert end_time > start_time, f'The second index {end_time} must be greater than the first index {start_time}'
        n_start = np.round(sampling_rate * start_time)
        n_end = np.round(sampling_rate * end_time)

        assert n_start < n_end, f'Start index must be smaller than end index'

        if start_time >= 0:
            ret = self._crop_array(array=array,
                                   n_start=n_start,
                                   n_end=n_end, crop_func=crop_func)
        else:
            # The start time is negative meaning we must first pad the array on the left
            padded_array = self._pad_array(array=array, pad_size=int(-n_start), pad_func=pad_func)
            new_start_time = 0
            new_end_time = end_time - start_time
            new_n_start = int(new_start_time * sampling_rate)
            new_end_end = int(new_end_time * sampling_rate)
            ret = self._crop_array(array=padded_array,
                                   n_start=new_n_start,
                                   n_end=new_end_end, crop_func=crop_func)
        return ret

    def get_adjusted_for_trigger_time(self, array, trigger_time=None, sampling_rate=None, crop_func=None,
                                      pad_func=None):

        sampling_rate = self.sampling_rate if sampling_rate is None else sampling_rate
        if trigger_time is None:
            trigger_time = self.trigger_time
        time_range = trigger_time
        return self.get_data_in_range(array, time_range, sampling_rate=sampling_rate, crop_func=None, pad_func=None)

    def set_attributes(self, **attributes):
        for attribute, value in attributes.items():
            setattr(self, attribute, value)


class BehavioralDataContainer(ModalityDataContainer):
    def __init__(self, data_dict: dict, timestamps_dict: dict, trigger_time: Union[float, None] = 0.0,
                 sampling_rate=None):
        super().__init__(data_dict, timestamps_dict)

    def get_raw_data(self):
        data_in = self.data_dict['video_']


class USVDataContainer(ModalityDataContainer):
    def __init__(self, data_dict, timestamps_dict: dict,
                 trigger_time: Union[float, None] = 0.0, sampling_rate=None):
        self.expand_timestamps_value = 0.05

        super().__init__(data_dict, timestamps_dict)

        self.trigger_time = trigger_time if trigger_time is not None else 0

        self.data = self.get_raw_data()
        if sampling_rate is None:
            self.sampling_rate = len(self.timestamps['FramesTimesInSec']) / (self.timestamps['FramesTimesInSec'][-1])

    def get_raw_data(self, labels=None):
        data_in = self.data_dict['audio_usv_data']
        return self._get_preprocess_data(data_in, labels).reshape(1, -1)

    def _get_empty_data(self):
        return np.zeros((1, len(self.timestamps['FramesTimesInSec'])))

    def _get_preprocess_data(self, df_in, labels=None):
        if labels is None:
            labels = ['USV-Auto']

        df_in = df_in.loc[df_in['Label'].isin(labels)]

        if len(df_in) == 0:
            return self._get_empty_data()
        return create_binary_list_with_ordered_data(
            df_in[['Begin_Time', 'End_Time']].values,
            self.timestamps['FramesTimesInSec'],
            dx=self.expand_timestamps_value)


class AudioDataContainer(ModalityDataContainer):
    def __init__(self,
                 data_dict,
                 timestamps_dict: dict,
                 sampling_rate: Union[float, int, None] = None,
                 trigger_time: Union[float, None] = 0.0,
                 ):
        super().__init__(
            data_dict=data_dict,
            timestamps_dict=timestamps_dict,
            sampling_rate=sampling_rate,
            trigger_time=trigger_time,
        )
        self.data = self.get_raw_data()
        self.set_sampling_rate(sampling_rate)

    def get_raw_data(self):
        sample = self.data_dict['audio_data'].reshape(1, -1)
        norm_sample = (sample - sample.mean()) / sample.std()
        return norm_sample

    def set_sampling_rate(self, sampling_rate=None):
        if sampling_rate is None:
            sampling_rate = self.data_dict['sample_rate']
        self.sampling_rate = sampling_rate


class ElectroDataContainer(ModalityDataContainer):
    def __init__(self,
                 data_dict,
                 timestamps_dict: dict,
                 sampling_rate: Union[float, int],
                 trigger_time: Union[float, None] = 0.0,
                 area_name_to_number_map_dict: dict = area_name_to_number_map,
                 area_number_to_name_map_dict: dict = area_number_to_name_map
                 ):

        super().__init__(
            data_dict=data_dict,
            timestamps_dict=timestamps_dict,
            sampling_rate=sampling_rate,
            trigger_time=trigger_time,
        )
        if sampling_rate is not None:
            assert sampling_rate > 0, f"Sampling rate must be a positive number but it was {sampling_rate}"

        self.area_name_to_number_map = area_name_to_number_map_dict
        self.area_number_to_name_map = area_number_to_name_map_dict
        self.probe_to_area_num = self.timestamps['ElectrodeVsRegisteredAreasNum']
        self.number_of_frames = len(self.timestamps['FramesTimesInSec'])

        self.nonactive_channels = None
        self.set_nonactove_channels()

        self.recorded_area_numbers = None
        self.set_recorded_area_numbers()

        self.recorded_area_names = None
        self.set_recorded_area_names()
        self.aggregation_method = 'mean'

    def get_raw_data(self, *args):
        raise NotImplemented

    def _get_empty_data(self):
        raise NotImplemented

    def set_recorded_area_numbers(self, data=None):
        if data is None:
            data_all = self.timestamps['ElectrodeVsRegisteredAreasNum']

            mask = np.isin(data_all[:, 0], self.nonactive_channels)
            filtered_arr = data_all[~mask]
            active_area_numbers = np.unique(filtered_arr[:, 1])
        else:
            active_area_numbers = data

        # make sure that the active area
        for n in active_area_numbers:
            if n not in self.area_number_to_name_map.keys():
                raise KeyError(
                    f'Area number {n} was not found in the list of areas: {self.area_number_to_name_map.keys()}')
        self.recorded_area_numbers = active_area_numbers

    def set_recorded_area_names(self):
        recorded_area_numbers = self.recorded_area_numbers
        self.recorded_area_names = [self.area_number_to_name_map[n]['name'] for n in recorded_area_numbers]

    def set_nonactove_channels(self, data=None):
        if data is None:
            data = self.data_dict['NonActiveElectroChannels']
        self.nonactive_channels = data

    def get_channel_by_name(self, area_name: Union[str, List[str]], method: Union[str, None] = None) -> np.ndarray:
        if method is None:
            method = self.aggregation_method

        # Check if the area was measured:
        # if area_name not in self.recorded_area_names:
        #     return self._get_empty_data()
        try:
            area_number = self.area_name_to_number_map[area_name]['number']
        except KeyError:
            valid_area_names = list(self.area_name_to_number_map.keys())
            raise KeyError(f"The specified area name must be one of {valid_area_names}, but it was {area_name}")

        return self.get_channel_by_number(area_number, method)

    def get_channel_by_number(self, area_number: int, method: Union[str, None] = None) -> np.ndarray:

        if method is None:
            method = self.aggregation_method
        active_prove_map = self.timestamps['ElectrodeVsRegisteredAreasNum']
        mat_inds = active_prove_map[active_prove_map[:, 1] == area_number, 0]
        if len(mat_inds) == 0:
            return self._get_empty_data()
        else:
            mat_inds = np.array(mat_inds, int) - 1

        return self.get_channel_by_index(mat_inds, method)

    def get_channel_by_index(self, mat_inds: Union[List[int], int], method: str = None) -> np.ndarray:

        if method is None:
            method = self.aggregation_method

        data = self.data
        data_slice = data[mat_inds, :]

        if data_slice.shape[0] > 1:
            if method == 'mean':
                data_slice = np.mean(data_slice, axis=0)
            elif method == 'and':
                data_slice = np.prod(data_slice)
            elif method == 'or':
                data_slice = np.any(data_slice, axis=0).astype(int)
            else:
                raise ValueError(f'Unknown aggregation method')
        data_slice = data_slice.reshape(1, -1)
        return data_slice


class LfpDataContainer(ElectroDataContainer):
    def __init__(self,
                 data_dict: dict,
                 timestamps_dict: dict,
                 trigger_time: Union[float, None] = 0.0,
                 sampling_rate: int = 5000):
        super().__init__(data_dict, timestamps_dict=timestamps_dict, trigger_time=trigger_time,
                         sampling_rate=sampling_rate)
        # Electrophysiology data
        self.data_dict = data_dict
        self.data_numel = self.data.shape[1]
        # self.coherence_config = None

    def _get_empty_data(self):
        return np.empty((0, self.data_numel))

    def get_raw_data(self):
        sample = self.data_dict['filteredDataLowSummary']
        # dc remove
        # sample = [remove_mains_hum(a, self.sampling_rate) for a in sample]

        # down sample
        original_sampling_rate = self.sampling_rate
        target_sampling_rate = 500
        # sample = [downsample_signal(a, original_sampling_rate, target_sampling_rate) for a in sample]
        sample = np.array(sample)
        self.sampling_rate = target_sampling_rate

        norm_sample = (sample - np.mean(sample, axis=1).shape) / np.std(sample, axis=1).shape
        return norm_sample


class LfpCoherenceDataContainer:
    def __init__(self, lfp_object: LfpDataContainer, area1: Union[str, None] = None, area2: Union[str, None] = None,
                 window_sec: float = 2, overlap_sec: float = 1.9, freq_range: Union[tuple, List, None] = None):
        self.lfp = lfp_object
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.freq_range = freq_range
        self.area1 = None
        self.area2 = None

    def get_data_in_range(self,
                          time_range: Union[tuple, List],
                          area1: str = None, area2: str = None,
                          window_size: Union[int, None] = None,
                          overlap_size: Union[int, None] = None,
                          freq_range: Union[tuple, List, None, str] = None) -> tuple[
        np.ndarray | tuple[np.ndarray, float | None], Any, Any]:

        if window_size is None:
            window_size = np.max([np.round(self.lfp.sampling_rate * self.window_sec), 1])

        if overlap_size is None:
            overlap_size = np.max([np.round(self.lfp.sampling_rate * self.overlap_sec), 1])

        if freq_range is None:
            freq_range = self.freq_range

        elif isinstance(str, freq_range):
            if freq_range == 'all':  # get max range 0-nyquist range
                freq_range = self.lfp.get_max_freq_range()
        sampling_rate = self.lfp.sampling_rate

        x1 = self.lfp.get_data_in_range(self.lfp.get_channel_by_name(area1), time_range)
        x2 = self.lfp.get_data_in_range(self.lfp.get_channel_by_name(area2), time_range)

        n_bins = x1.shape[1]
        if window_size >= n_bins:
            window_size = np.max([n_bins//2, 2])
            overlap_size = np.max([int(window_size*0.8), 1])


        freq, coh = compute_time_coherence(x=x1, y=x2, segment_length=window_size, overlap_factor=0.95,
                                           sampling_rate=sampling_rate, window_size=window_size // 2,
                                           overlap_size=overlap_size // 2)
        coh = coh.transpose()
        time_vec = np.linspace(time_range[0], time_range[1], coh.shape[1])
        # return calculate_coherence(x1, x2, sampling_rate, window_size, freq_range=freq_range)
        return time_vec, freq, coh


class MultiSpikeDataContainer(ElectroDataContainer):
    def __init__(self,
                 data_dict: dict,
                 timestamps_dict: dict,
                 trigger_time: Union[float, None] = 0.0,
                 sampling_rate: Union[int, None] = None):
        super().__init__(data_dict, timestamps_dict=timestamps_dict, trigger_time=trigger_time,
                         sampling_rate=sampling_rate)
        # Electrophysiology data

        self.data_dict = data_dict
        self.data_numel = len(self.timestamps['FramesTimesInSec'])  # Get number of frames
        if sampling_rate is None:
            self.sampling_rate = len(self.timestamps['FramesTimesInSec']) / (self.timestamps['FramesTimesInSec'][-1])
        self.aggregation_method = 'or'

    def get_raw_data(self):
        data_in = self.data_dict['SpikeTimes']

        # reshape the array
        if not isinstance(data_in, list):
            data_in_temp = []
            for data in data_in[0]:
                if data is None:
                    ret = None
                else:
                    ret = data.reshape(-1, )
                data_in_temp.append(ret)
            data_in = data_in_temp
            self.data_dict['SpikeTimes'] = data_in

        return self._get_preprocess_data(data_in)

    def _get_empty_data(self):
        return np.zeros((1, len(self.timestamps['FramesTimesInSec'])))

    def _get_preprocess_data(self, data_in):
        ret = []

        for channel in data_in:
            if channel is None:
                np_array = self._get_empty_data()
            else:
                data = convert_time_stamps_to_frame_numbers(channel, self.timestamps['FramesTimesInSec'])
                np_array = convert_to_numpy_array(timestamps=data,
                                                  num_timestamps=len(self.timestamps['FramesTimesInSec'])).reshape(1,
                                                                                                                   -1)
            ret.append(np_array)
        return np.array(ret)


class PoseDataContainer(ModalityDataContainer):
    def __init__(self, data_dict, timestamps_dict: dict,
                 trigger_time: Union[float, None] = 0.0, sampling_rate=None):
        self.expand_timestamps_value = 0.05

        super().__init__(data_dict, timestamps_dict)
        if sampling_rate is None:
            self.sampling_rate = len(self.timestamps['FramesTimesInSec']) / (self.timestamps['FramesTimesInSec'][-1])
        else:
            self.sampling_rate = sampling_rate
        self.set_stimulus_insertion_time(self.timestamps['StimulusInsertionFrame'], self.sampling_rate)
        if trigger_time is None:
            self.trigger_time = self.insertion_time + self.timestamps['StartTriggerTimeInSec']
        else:
            self.trigger_time = trigger_time


        self.data = self.get_raw_data()
        self.time_dim = 0
        self._crop_func = self._get_crop_func
        self._pad_func = self._get_pad_func




    def get_raw_data(self, labels=None):
        data_in = self.data_dict['pose_data']
        return self._get_preprocess_data(data_in)

    def _get_empty_data(self, n):
        data_shape = list(self.data.shape)
        data_shape[self.time_dim] = n
        return np.full(data_shape,np.nan )

    def _get_preprocess_data(self, df_in, labels=None):
        if len(df_in) == 0:
            return self._get_empty_data()
        df_in = fix_df_col_names(df_in)
        return pose_df_to_matrix(df_in)

    # function to retrieve data

    def _get_crop_func(self, x, n_start, n_end):
        max_ind = self.get_max_range_inds()

        n_end = int(np.min([n_end, max_ind]))
        n_start = int(np.min([n_start, n_end]))
        max_pose_ind = self.data.shape[self.time_dim]

        # the requested range exceeds the avaialble pose
        if n_start >= max_pose_ind:
            return self._get_empty_data(n=(n_end-n_start))

        elif n_start < max_pose_ind <= n_end:
            pad_size = n_end - max_pose_ind
            x = self._pad_func(x, pad_size, pad_end=True)


        return x[n_start:n_end, :, :, :]

    def _get_pad_func(self, array, pad_size, pad_end=False):
        if pad_end:
            pad_tuple = (0, pad_size)
        else:
            pad_tuple = (pad_size, 0)
        pad_width = (pad_tuple, (0, 0), (0, 0), (0, 0))

        # Pad the array with NaN values
        padded_array = np.pad(array, pad_width, mode='constant', constant_values=np.nan)
        return padded_array









class SocialExpSession(SocialExpSessionBase):
    def __init__(self, dict_in: dict, modalities_to_load: list):
        super().__init__()


        self.data_path_dict = dict_in
        self.modality_options = ['lfp', 'lfp_coherence', 'multispike', 'audio', 'usv', 'pose_data']  # types of modalities
        self.modalities_to_load = modalities_to_load if modalities_to_load is not None else self.modality_options
        assert [a in self.modality_options for a in modalities_to_load], f'Invalid modality to load, must me in {self.modality_options} but is was {modalities_to_load} '
        # ================== Set session id parameters =========
        self.rat_num = self.data_path_dict['rat_number']
        self.day_num = self.data_path_dict['day_number']
        self.subject_sex = self.data_path_dict.get('subject_sex')
        self.stimulus_sex = self.data_path_dict.get('stimulus_sex')
        self.stimulus_age = self.data_path_dict.get('stimulus_age')
        self.sociability = self.data_path_dict.get('sociability')
        self.probe_name = self.data_path_dict.get('probe_number')
        self.paradigm = self.data_path_dict.get('paradigm')
        self.info_table = self.get_info_table()

        # ==================== load data =====================

        self.modalities = {}

        # Set timestamp parameters
        self.timestamps = self.load_files('Timestemps')[0]
        # when loading using scipy the time stamp array must be reshaped
        self.timestamps['FramesTimesInSec'] = self.timestamps['FramesTimesInSec'].reshape(-1, )
        self.set_trigger_time(self.timestamps['StartTriggerTimeInSec'])
        self.frame_rate = 1 / np.mean(
            self.timestamps['FramesTimesInSec'][1:] - self.timestamps['FramesTimesInSec'][:-1])
        self.stimulus_insertion_time = (self.timestamps['StimulusInsertionFrame'].reshape(-1, ) / self.frame_rate)[
                                           0] - self.trigger_time  # check if we need to subtract trigger time
        self.stimulus_removal_time = (self.timestamps['StimulusRemovalFrame'].reshape(-1, ) / self.frame_rate)[
                                         0] - self.trigger_time  # check if we need to subtract trigger time

        if 'usv_data' in self.modalities_to_load:
            # Load audio vocalizations analysis file
            self.modalities['usv_data'] = USVDataContainer(self.load_files('audio excel files')[0],
                                                           self.timestamps, self.trigger_time)
        if 'lfp' in self.modalities_to_load:
        # Load LFP data
            self.modalities['lfp'] = LfpDataContainer(self.load_files('LFP')[0],
                                                      self.timestamps,self.trigger_time)

        if 'lfp_coherence' in self.modalities_to_load:
            # Set coherence object
            self.modalities['lfp_coherence'] = LfpCoherenceDataContainer(self.modalities['lfp'], window_sec=2)

        if 'multispikes' in self.modalities_to_load:
            # Load multispike data
            self.modalities['multispikes'] = MultiSpikeDataContainer(self.load_files('Multispikes')[0],
                                                                     self.timestamps, self.trigger_time)
        if 'audio' in self.modalities_to_load:
            # Load raw audio data
            self.modalities['audio'] = AudioDataContainer(
                self.load_files('Audio-wav files', is_audio=True)[0],
                timestamps_dict=self.timestamps,
                trigger_time=None)

        if 'pose_data' in self.modalities_to_load:
            # Load pose data
            self.modalities['pose_data'] = PoseDataContainer(
                self.load_files('video_pose_estimation')[0],
                timestamps_dict=self.timestamps,
                trigger_time=None
            )


        # Load behavioral data
        # self.behavioral_data = BehavioralDataContainer(
        #     self.load_files('video files')[0],
        #     self.timestamps, self.trigger_time
        # )

    def get_max_range(self):
        return self.timestamps['FramesTimesInSec'][-1]

    def load_data_in_range(self, time_range=(390, 400), area_name='CeA', modalities_to_load=None):

        if modalities_to_load is None or modalities_to_load == 'all':
            modalities_to_load = self.modality_options
        # modality_options = ['lfp', 'multispike', 'audio', 'usv']

        if isinstance(area_name, str):
            if area_name == 'all':
                area_name = self.modalities['lfp'].recorded_area_names
            else:
                area_name = [area_name]

        dict_to_return = {}

        for modality in modalities_to_load:
            if modality == 'lfp':
                vec_dict = {}
                for area in area_name:
                    vec_dict[area] = self.modalities['lfp'].get_data_in_range(
                        self.modalities['lfp'].get_channel_by_name(area),
                        time_range)
                dict_to_return['lfp'] = vec_dict

            elif modality == 'lfp_coherence':
                vec_dict = {}
                # self.modalities['lfp_coherence']['areas']
                for area_pair in area_name:
                    area1, area2 = area_pair
                    t,f,mat = self.modalities['lfp_coherence'].get_data_in_range(
                        area1=area1, area2=area2,time_range=time_range, window_size=2000, overlap_size=1980)

                    vec_dict[tuple(area_pair)] = {'time': t, 'freq':f, 'data':mat}
                dict_to_return['lfp_coherence'] = vec_dict

            elif modality == 'multispike':
                vec_dict = {}
                for area in area_name:
                    vec_dict[area] = self.modalities['multispikes'].get_data_in_range(
                        self.modalities['multispikes'].get_channel_by_name(area, method='or'),
                        time_range)
                dict_to_return['multispike'] = vec_dict

            elif modality == 'audio':
                vec = self.modalities['audio'].get_data_in_range(self.modalities['audio'].data, time_range)
                dict_to_return['audio'] = vec

            elif modality == 'usv':
                vec = self.modalities['usv_data'].get_data_in_range(self.modalities['usv_data'].get_raw_data(),
                                                                    time_range)
                dict_to_return['usv'] = vec
            else:
                raise KeyError(f'Invalid key {modality}, must be on of {self.modality_options}')

        # vec1 = butter_bandpass_filter(data=vec1, lowcut=4.0, highcut=30.0, fs=s_session.lfp.sampling_rate, order=3)

        timestamp_array = self.modalities['multispikes'].get_data_in_range(
            self.timestamps['FramesTimesInSec'].reshape(1, -1),
            time_range)

        return timestamp_array, dict_to_return

    def get_info_table(self):
        return pd.DataFrame.from_dict({'Rat number': [self.rat_num],
                                       'Day number': [self.day_num],
                                       'Probe number': [self.probe_name],
                                       'Paradigm': [self.paradigm],
                                       'Subject_sex': [self.subject_sex],
                                       'Stimulus_sex': [self.stimulus_sex],
                                       'Stimulus_age': [self.stimulus_age],
                                       'Sociability': [self.sociability]})

    def __str__(self):

        # str_ret = f'Rat:{self.rat_num}' \
        #           f'Day:{self.day_num}' \
        #           f'Probe:{self.probe_name}' \
        #           f'Paradigm{self.paradigm}' \
        #           f'Subject_sex: {self.subject_sex}' \
        #           f'Stimulus_sex: {self.stimulus_sex}' \
        #           f'Stimulus_age: {self.stimulus_age}' \
        #           f'Sociability:{self.sociability}'
        return self.info_table.transpose().to_string()







if __name__ == '__main__':
    # Get list of file names
    # path_to_files = osp.join('..', 'assets', 'session to file mapping reordered.xlsx')
    path_to_files = FILES_MAPPING_PATH

    df = pd.read_excel(path_to_files)
    df = df.loc[df['paradigm'].isin(['free', 'chamber'])]
    first_line = df.iloc[5, :]

    first_line = dict(first_line)
    print(first_line.keys())

    s_session = SocialExpSession(first_line, modalities_to_load=['pose_data'])
    # vec1 = s_session.lfp.get_adjusted_for_trigger_time(s_session.lfp.get_channel_by_name('CeA'))
    # vec2 = s_session.multispikes.get_adjusted_for_trigger_time(
    #     s_session.multispikes.get_channel_by_name('CeA', method='or'))
    # t, f, coh = s_session.modalities['lfp_coherence'].get_data_in_range(area1='EA', area2='MeD', time_range=(300, 315))
    # plt.pcolormesh(t, f, coh)
    ret = s_session.modalities['pose_data'].get_data_in_range(s_session.modalities['pose_data'].get_raw_data(),  time_range=(317, 319))
    print(ret.shape)
    # s_session.load_data_in_range(time_range=(300, 350), area_name='all')
    print(s_session)

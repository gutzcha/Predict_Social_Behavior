from typing import List, Union, Tuple
from utils.get_project_root import get_project_root
from utils.read_matlab_file import read_matlab
import os
import os.path as osp
import pandas as pd
import numpy as np
import soundfile as sf
from utils.consts import area_name_to_number_map, area_number_to_name_map
from utils.helper_functions import downsample_signal, remove_mains_hum
from scipy.sparse import csr_matrix
import bisect


# Static functions
def create_binary_list_with_ordered_data(time_ranges: Union[List, np.ndarray], timestamps: Union[List, np.ndarray],
                                         dx: float = 0.0) -> np.ndarray:
    binary_list = np.zeros(len(timestamps), dtype=int)
    ranges_left = [start_time - dx for start_time, _ in time_ranges]

    for i, timestamp in enumerate(timestamps):
        index = bisect.bisect_left(ranges_left, timestamp)
        if index > 0 and timestamp <= time_ranges[index - 1][1] + dx:
            binary_list[i] = 1

    return binary_list


def create_binary_array(time_ranges, sampling_rate, total_length_seconds):
    total_samples = int(sampling_rate * total_length_seconds)
    binary_array = np.zeros(total_samples, dtype=int)

    for start_time, end_time in time_ranges:
        start_index = int(start_time * sampling_rate)
        end_index = int(end_time * sampling_rate)

        # Clip the indices to ensure they are within the array's bounds
        start_index = max(start_index, 0)
        end_index = min(end_index, total_samples)

        binary_array[start_index:end_index] = 1

    return binary_array


def convert_to_sparse_array(timestamps, num_timestamps):
    # Create an empty matrix with the desired shape
    sparse_array = csr_matrix((1, num_timestamps), dtype=np.int8)

    # Iterate over the timestamps and set the corresponding element to 1
    for timestamp in timestamps:
        sparse_array[0, timestamp] = 1

    return sparse_array


def convert_to_numpy_array(timestamps, num_timestamps):
    # Create an empty NumPy array with zeros
    numpy_array = np.zeros(num_timestamps, dtype=int)

    # Set the corresponding elements to 1 based on the timestamps
    numpy_array[timestamps] = 1

    return numpy_array


def convert_time_stamps_to_frame_numbers(first_array, second_array, dx=0.0001):
    indices = []
    for num in first_array:
        lower_bound = num - dx
        upper_bound = num + dx

        # Find the indices where the element from the first array would fit
        lower_idx = np.searchsorted(second_array, lower_bound, side='left')
        upper_idx = np.searchsorted(second_array, upper_bound, side='right')

        # Add the indices to the list
        indices.extend(range(lower_idx, upper_idx))

    return indices


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
        elif file_path.lower().endswith('xlsx'):
            return [{'audio_usv_data': pd.read_excel(file_path)}]
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
        self.insertion_time = insertion_frame * frame_rate


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

        self.timestamps = timestamps_dict

        self.data_dict = data_dict

        self.sampling_rate = sampling_rate

        self.trigger_time = 0 if trigger_time is None else trigger_time

        self.data = self.get_raw_data()

    def get_time_from_ind(self, ind):
        return ind / self.sampling_rate + self.trigger_time

    def get_max_range(self):  # TODO Check this later
        return self.timestamps[-1]

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
            return x[:, int(np.round(n_start)):int(np.round(n_end))]

        if n_end is None or n_end > array.shape[-1]:
            n_end = array.shape[-1]
        if crop_func is None:
            crop_func = default_crop_func
        return crop_func(x=array, n_start=n_start, n_end=n_end)

    def get_data_in_range(self, array,
                          time_range: Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
                          sampling_rate=None, crop_func=None,
                          pad_func=None):

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

        if start_time >= 0:
            ret = self._crop_array(array=array,
                                   n_start=sampling_rate * start_time,
                                   n_end=sampling_rate * end_time, crop_func=crop_func)
        else:
            # The start time is negative meaning we must first pad the array on the left
            padded_array = self._pad_array(array=array, pad_size=-sampling_rate * start_time, pad_func=pad_func)
            new_start_time = 0
            new_end_time = end_time - start_time
            ret = self._crop_array(array=padded_array,
                                   n_start=sampling_rate * new_start_time,
                                   n_end=sampling_rate * new_end_time, crop_func=crop_func)
        return ret

    def get_adjusted_for_trigger_time(self, array, trigger_time=None, sampling_rate=None, crop_func=None,
                                      pad_func=None):

        sampling_rate = self.sampling_rate if sampling_rate is None else sampling_rate
        if trigger_time is None:
            trigger_time = self.trigger_time
        time_range = trigger_time
        return self.get_data_in_range(array, time_range, sampling_rate=sampling_rate, crop_func=None, pad_func=None)


class BehavioralDataContainer(ModalityDataContainer):
    def __init__(self,data_dict: dict, timestamps_dict: dict, trigger_time: Union[float, None] = 0.0, sampling_rate=None):
        super().__init__(data_dict, timestamps_dict)

    def get_raw_data(self):
        data_in = self.data_dict['video_']

class USVDataContainer(ModalityDataContainer):
    def __init__(self, data_dict, timestamps_dict: dict, trigger_time: Union[float, None] = 0.0, sampling_rate=None):
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


class SocialExpSession(SocialExpSessionBase):
    def __init__(self, dict_in: dict):
        super().__init__()

        self.data_path_dict = dict_in
        self.modality_options = ['lfp', 'multispike', 'audio', 'usv']  # types of modalities

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

        # Load audio vocalizations analysis file
        self.usv_data = USVDataContainer(self.load_files('audio excel files')[0],
                                         self.timestamps, self.trigger_time)

        # Load LFP data
        self.lfp = LfpDataContainer(self.load_files('LFP')[0],
                                    self.timestamps,
                                    self.trigger_time)
        # Load multispike data
        self.multispikes = MultiSpikeDataContainer(self.load_files('Multispikes')[0],
                                                   self.timestamps,
                                                   self.trigger_time)
        # Load raw audio data
        self.audio = AudioDataContainer(
            self.load_files('Audio-wav files', is_audio=True)[0],
            timestamps_dict=self.timestamps,
            trigger_time=None)

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
                area_name = self.lfp.recorded_area_names
            else:
                area_name = [area_name]

        dict_to_return = {}

        for modality in modalities_to_load:
            if modality == 'lfp':
                vec_dict = {}
                for area in area_name:
                    vec_dict[area] = self.lfp.get_data_in_range(self.lfp.get_channel_by_name(area),
                                                                time_range)
                dict_to_return['lfp'] = vec_dict

            elif modality == 'multispike':
                vec_dict = {}
                for area in area_name:
                    vec_dict[area] = self.multispikes.get_data_in_range(
                        self.multispikes.get_channel_by_name(area, method='or'),
                        time_range)
                dict_to_return['multispike'] = vec_dict
            elif modality == 'audio':
                vec = self.audio.get_data_in_range(self.audio.data, time_range)
                dict_to_return['audio'] = vec

            elif modality == 'usv':
                vec = self.usv_data.get_data_in_range(self.usv_data.get_raw_data(), time_range)
                dict_to_return['usv'] = vec
            else:
                raise KeyError(f'Invalid key {modality}, must be on of {self.modality_options}')

        # vec1 = butter_bandpass_filter(data=vec1, lowcut=4.0, highcut=30.0, fs=s_session.lfp.sampling_rate, order=3)

        timestamp_array = self.multispikes.get_data_in_range(
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
    path_to_files = osp.join('..', 'assets', 'file_to_session_mapping_20230725.xlsx')

    df = pd.read_excel(path_to_files)
    df = df.loc[df['paradigm'].isin(['free', 'chamber'])]
    first_line = df.iloc[5, :]

    first_line = dict(first_line)
    print(first_line.keys())

    s_session = SocialExpSession(first_line)
    # vec1 = s_session.lfp.get_adjusted_for_trigger_time(s_session.lfp.get_channel_by_name('CeA'))
    # vec2 = s_session.multispikes.get_adjusted_for_trigger_time(
    #     s_session.multispikes.get_channel_by_name('CeA', method='or'))
    s_session.load_data_in_range(time_range=(300,350), area_name='all')
    print(s_session)

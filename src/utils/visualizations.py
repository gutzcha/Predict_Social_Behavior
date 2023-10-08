import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from main.session_class import SocialExpSession
import pandas as pd
from helper_functions import plot_spectrogram
from utils.consts import FILES_MAPPING_PATH

#
# def plot_interpolated_arrays(arrays, timestamp_array, names=None):
#     # Create a figure with subplots
#     fig, axs = plt.subplots(len(arrays), 1, sharex='all')
#     if not names is None:
#         assert len(names) == len(arrays),'numer of names must match number of arrays'
#     else:
#         names = [f'Array {i}' for i in range(len(arrays))]
#     # Iterate over the arrays and plot in the respective subplots
#     for i, array in enumerate(arrays):
#         length = array.shape[1]
#         adjusted_timestamps = np.linspace(timestamp_array[0, 0], timestamp_array[0, -1], length)
#         axs[i].plot(adjusted_timestamps, array[0, :])
#         axs[i].set_ylabel(names[i])
#
#     axs[-1].set_xlabel('Time')
#
#     # Display the figure
#     plt.show()
#     return axs

def load_example(ind=0):
    # path_to_files = osp.join('..', 'assets', 'session to file mapping reordered.xlsx')
    path_to_files = FILES_MAPPING_PATH


    df = pd.read_excel(path_to_files)
    df = df.loc[df['paradigm'].isin(['free', 'chamber'])]
    first_line = df.iloc[ind, :]

    first_line = dict(first_line)
    # print(first_line.keys())

    s_session = SocialExpSession(first_line)
    return s_session


def plot_spectogram(x, fs):
    NFFT = 64  # the length of the windowing segments

    Pxx, freqs, t, im = plt.specgram(x, NFFT=NFFT, Fs=fs, noverlap=32, detrend='mean')
    # The `specgram` method returns 4 objects. They are:
    # - Pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the .image.AxesImage instance representing the data in the plot
    plt.show()
    return Pxx, freqs, t, im


def load_data_in_range(s_session, time_range=(390, 400), area_name='CeA'):
    vec1 = s_session.lfp.get_data_in_range(s_session.lfp.get_channel_by_name(area_name), time_range)
    # vec1 = butter_bandpass_filter(data=vec1, lowcut=4.0, highcut=30.0, fs=s_session.lfp.sampling_rate, order=3)
    vec2 = s_session.multispikes.get_data_in_range(s_session.multispikes.get_channel_by_name(area_name, method='or'),
                                                   time_range)
    vec3 = s_session.audio.get_data_in_range(s_session.audio.data, time_range)
    vec4 = s_session.usv_data.get_data_in_range(s_session.usv_data.get_raw_data(), time_range)
    timestamp_array = s_session.multispikes.get_data_in_range(s_session.timestamps['FramesTimesInSec'].reshape(1, -1),
                                                              time_range)
    return timestamp_array, [vec1, vec2, vec3, vec4], ['lfp', 'multispike', 'audio', 'usv']


import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display


# Import your functions (load_data_in_range, plot_interpolated_arrays) here
# Ensure that the functions are appropriately defined with necessary parameters.

class PlotArrays():
    def __init__(self, time_range_start=0, time_range_end=10):
        self.time_range_start = time_range_start
        self.time_range_end = time_range_end
        self.step_size = 1
        self.s_session = load_example(ind=4)

        time_range_start_slider = widgets.FloatSlider(min=0, max=100, step=0.1, value=self.time_range_start,
                                                      description='Start Time:')
        time_range_end_slider = widgets.FloatSlider(min=0, max=100, step=0.1, value=self.time_range_end,
                                                    description='End Time:')
        step_size_box = widgets.FloatText(value=self.step_size, description='Step Size:')
        forward_button = widgets.Button(description='Forward')
        backward_button = widgets.Button(description='Backward')

        # Set up event handlers
        forward_button.on_click(self.on_forward_button_click)
        backward_button.on_click(self.on_backward_button_click)
        step_size_box.observe(self.on_step_size_change, names='value')

        # Display interactive elements
        display(
            widgets.VBox(
                [time_range_start_slider, time_range_end_slider, step_size_box, forward_button, backward_button]))

        # Update the plot with initial parameters
        self.update_plot()

    def update_plot(self):
        # Load data with updated time_range
        timestamp_array, vector_list, vector_names = load_data_in_range(self.s_session, time_range=(
        self.time_range_start, self.time_range_end), area_name='EA')

        # Create the plot with updated data
        plt.figure(figsize=(10, 6))
        spectogram_dict_list = [{'index': i, 'frequency_range': (1, 50), 'nperseg': 256, 'noverlap': 200,
                                 'sampling_rate': self.s_session.lfp.sampling_rate} for i in range(len(vector_list))]
        plot_interpolated_arrays(vector_list, timestamp_array, vector_names, spectogram_dict_list=spectogram_dict_list)
        plt.show()

    def on_forward_button_click(self, b):
        self.time_range_start += self.step_size
        self.time_range_end += self.step_size
        self.update_plot()

    def on_backward_button_click(self, b):
        self.time_range_start -= self.step_size
        self.time_range_end -= self.step_size
        self.update_plot()

    def on_step_size_change(self, change):

        self.step_size = change['new']


if __name__ == '__main__':
    # s_session = load_example()
    # timestamp_array, vector_list, vector_names = load_data_in_range(s_session, time_range=(390, 400), area_name='CeA')
    # # plot_interpolated_arrays(vector_list, timestamp_array, vector_names)
    # # plot_spectrogram(signal=vector_list[0][0], sampling_rate=s_session.lfp.sampling_rate, freq_range=(1, 50),
    # #                  window='hann', nperseg=256, noverlap=None, ax=None)
    #
    # print(t + 390)
    # import matplotlib.pyplot as plt
    # from visualizations import load_example, load_data_in_range  # plot_interpolated_arrays
    from helper_functions import plot_spectrogram, plot_interpolated_arrays

    # # %%
    #
    # s_session = load_example(ind=4)
    # # %%
    # timestamp_array, vector_list, vector_names = load_data_in_range(s_session, time_range=(390, 400), area_name='EA')
    #
    # # %%
    # # vector_list[0]
    #
    # # %%
    #
    # spectogram_dict_list = [{'index': 0, 'frequency_range': (1, 50), 'nperseg': 256, 'noverlap': 200,
    #                          'sampling_rate': s_session.lfp.sampling_rate},
    #                         {'index': 2, 'frequency_range': (20000, 80000), 'nperseg': 256, 'noverlap': 200,
    #                          'sampling_rate': s_session.audio.sampling_rate}
    #                         ]
    # plot_interpolated_arrays(vector_list, timestamp_array, vector_names,
    #                                 spectogram_dict_list=spectogram_dict_list)
    # # plot_spectrogram(signal=vector_list[0][0],
    # #                  sampling_rate=s_session.lfp.sampling_rate,
    # #                  freq_range=(1,50), window='hann', nperseg=256, noverlap=None,axes=axis[0])
    # plt.show()



    # Create interactive elements

    PlotArrays()
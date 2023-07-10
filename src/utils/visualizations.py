import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from src.main.session_class import SocialExpSession
import pandas as pd

def plot_interpolated_arrays(arrays, timestamp_array):
    # Create a figure with subplots
    fig, axs = plt.subplots(len(arrays), 1, sharex='all')

    # Iterate over the arrays and plot in the respective subplots
    for i, array in enumerate(arrays):
        length = array.shape[1]
        adjusted_timestamps = np.linspace(timestamp_array[0, 0], timestamp_array[0, -1], length)
        axs[i].plot(adjusted_timestamps, array[0, :])
        axs[i].set_ylabel(f'Array {i+1}')

    axs[-1].set_xlabel('Time')

    # Display the figure
    plt.show()

def plot_sample():
    path_to_files = osp.join('..', 'assets', 'session to file mapping reordered.xlsx')
    df = pd.read_excel(path_to_files)
    df = df.loc[df['paradigm'].isin(['free', 'chamber'])]
    first_line = df.iloc[0, :]

    first_line = dict(first_line)
    # print(first_line.keys())

    s_session = SocialExpSession(first_line)
    time_range = (350, 400)
    vec1 = s_session.lfp.get_data_in_range(s_session.lfp.get_channel_by_name('CeA'), time_range)
    vec2 = s_session.multispikes.get_data_in_range(s_session.multispikes.get_channel_by_name('CeA', method='or'), time_range)
    vec3 = s_session.audio.get_data_in_range(s_session.audio.data, time_range)

    timestamp_array = s_session.multispikes.get_data_in_range(s_session.timestamps['FramesTimesInSec'].reshape(1,-1), time_range)

    plot_interpolated_arrays([vec1, vec2, vec3], timestamp_array)


if __name__ == '__main__':
    plot_sample()
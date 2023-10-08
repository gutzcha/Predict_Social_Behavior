from scipy.signal import butter, lfilter, spectrogram, filtfilt, welch, coherence
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import pywt
from scipy.sparse import csr_matrix
from typing import List, Union, Tuple
import bisect
from sklearn.decomposition import FastICA
from utils.consts import NODES2IND, EDGES, NODES
from scipy.signal import savgol_filter


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


def compute_time_coherence(x, y, segment_length, overlap_factor, sampling_rate, window_size, overlap_size,
                           derivative_flag=True, use_ica=False):
    coherence_vector = []
    if use_ica:
        input_signals = np.vstack((x, y)).T
        ica = FastICA(n_components=2, random_state=0)
        independent_sources = ica.fit_transform(input_signals)
        x = independent_sources[:, 0].reshape((1, -1))
        y = independent_sources[:, 0].reshape((1, -1))
        # if derivative_flag:
    #     x = np.diff(x)
    #     y = np.diff(y)
    #
    #     np.insert(x, 0, 0)
    #     np.insert(y, 0, 0)
    freq = []
    for i in range(0, x.shape[1] - segment_length + 1, int(segment_length * (1 - overlap_factor))):
        x_seg = x[0][i:i + segment_length]
        y_seg = y[0][i:i + segment_length]

        freq, coh = coherence(x_seg, y_seg, fs=sampling_rate, window='hann', nperseg=window_size, noverlap=overlap_size,
                              nfft=None, detrend='constant', axis=-1)
        coherence_vector.append(coh)
    coherence_vector = np.array(coherence_vector)
    # if derivative_flag:
    #     coherence_vector = np.diff(coherence_vector, axis=1)
    #     coherence_vector = np.hstack([coherence_vector, coherence_vector[:, -1][:, np.newaxis]])
    # coherence_vector = np.log(coherence_vector+0.00001)
    return freq, coherence_vector


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def continuous_wavelet_transform(data, fs, f0, fn, num_frequencies=100, wavelet='morl'):
    """
    Perform Continuous Wavelet Transform (CWT) on the input signal data.

    Parameters:
        data (numpy.ndarray): 1-dimensional input signal data of shape (n, ).
        fs (float): Sampling rate of the input signal (in Hz).
        f0 (float): Lower frequency of the range for CWT.
        fn (float): Upper frequency of the range for CWT (should be smaller than fs/2).
        num_frequencies (int, optional): Number of frequencies in the range. Default is 100.
        wavelet (str, optional): Wavelet function to be used. Default is 'morl'.

    Returns:
        time (numpy.ndarray): Time vector corresponding to the input signal data.
        frequencies (numpy.ndarray): Frequencies vector for CWT within the specified range.
        cwt_matrix (numpy.ndarray): 2D matrix of Continuous Wavelet Transform.
    """
    time = np.arange(0, len(data)) / fs

    # Define the frequencies range for CWT
    frequencies = np.linspace(f0, fn, num_frequencies)

    # Perform the Continuous Wavelet Transform for each frequency in the range
    cwt_matrix = np.zeros((num_frequencies, len(data)), dtype=complex)

    for i, freq in enumerate(frequencies):
        scales = fs / (freq * pywt.scale2frequency(wavelet, 1))
        coefficients, _ = pywt.cwt(data, scales, wavelet)
        cwt_matrix[i, :] = coefficients

    return time, frequencies, cwt_matrix


def plot_spectrogram(signal_in: object, sampling_rate: object, freq_range: object = None, window: object = 'hann',
                     nperseg: object = 256, noverlap: object = None,
                     axes: object = None,
                     show_flag: object = True) -> object:
    """
    Plot the spectrogram of a specific frequency range from the input signal.

    Parameters:
        signal (numpy array): 1D input signal
        sampling_rate (float): Sampling rate of the input signal (in Hz)
        freq_range (tuple or None, optional): Tuple containing the lower and upper frequency bounds of the range (in Hz).
                                              If None, the entire frequency range will be plotted. Defaults to None.
        window (str, optional): Type of window to use in the spectrogram computation. Defaults to 'hann'.
        nperseg (int, optional): Number of data points used in each segment for spectrogram computation. Defaults to 256.
        noverlap (int, optional): Number of points to overlap between segments. If None, noverlap = nperseg // 8. Defaults to None.
        axes (matplotlib.axes.Axes or None, optional): Specific axes or subplot to place the plot. If None, a new plot will be created. Defaults to None.
        show_flag (bool, optional): plot the spectrogram or return the values

    Returns:
        None (displays the spectrogram plot using matplotlib)
    """
    try:
        f_orig, t, Sxx = spectrogram(signal_in, fs=sampling_rate, window=window, nperseg=nperseg, noverlap=noverlap)
    except ValueError:
        Warning(f'Invalid spectrogram values, setting parameters to default')
        f_orig, t, Sxx = spectrogram(signal_in, fs=sampling_rate, window=window, nperseg=None, noverlap=None)

    if freq_range is not None:
        freq_mask = (f_orig >= freq_range[0]) & (f_orig <= freq_range[1])
        Sxx_filtered = Sxx[freq_mask]
        f = f_orig[freq_mask]
    else:
        Sxx_filtered = Sxx

    if not show_flag:
        return f, t, Sxx_filtered, f_orig

    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    ax.pcolormesh(t, f, 10 * np.log10(np.abs(Sxx_filtered) + 0.0001), shading='gouraud')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Spectrogram')

    if axes is None:
        plt.colorbar(label='Power Spectral Density (dB/Hz)')
        plt.show()


def plot_interpolated_arrays(arrays, timestamp_array, names=None, spectogram_dict_list=None, show_flag=True, fig=None,
                             axs=None):
    # Create a figure with subplots
    if show_flag:
        fig, axs = plt.subplots(len(arrays) + len(spectogram_dict_list), 1, sharex='all', gridspec_kw={'hspace': 0.5})

    if not names is None:
        assert len(names) == len(arrays), 'number of names must match number of arrays'
    else:
        names = [f'Array {i}' for i in range(len(arrays) + len(spectogram_dict_list))]

    # Iterate over the arrays and plot in the respective subplots
    i = 0
    k = 0
    for array, name in zip(arrays, names):
        length = array.shape[1]
        adjusted_timestamps = np.linspace(timestamp_array[0, 0], timestamp_array[0, -1], length)
        axs[i + k].plot(adjusted_timestamps, array[0, :])
        axs[i + k].set_ylabel(name)

        # Check if spectrogram_dict_list is provided and if the current index exists in the list
        if spectogram_dict_list is not None and i in [item['index'] for item in spectogram_dict_list]:
            spec_dict = next((item for item in spectogram_dict_list if item['index'] == i), None)
            if spec_dict is not None:
                k += 1
                freq_range = spec_dict.get('frequency_range', None)
                nperseg = spec_dict.get('nperseg', 256)
                noverlap = spec_dict.get('noverlap', 200)
                sampling_rate = spec_dict.get('sampling_rate', 1.0 / (timestamp_array[0, 1] - timestamp_array[0, 0]))
                # Calculate spectrogram using the provided data and parameters
                # Use 'spec_params' key to access additional parameters required for spectrogram
                f, t, spectrogram_data = plot_spectrogram(array[0], sampling_rate, freq_range=freq_range, window='hann',
                                                          nperseg=nperseg, noverlap=noverlap,
                                                          axes=None,
                                                          show_flag=False)
                adjusted_timestamps = np.linspace(timestamp_array[0, 0], timestamp_array[0, -1], len(t))

                # spectrogram_data = calculate_spectrogram(array, timestamp_array, freq_range,
                #                                          spec_dict.get('spec_params', None))
                # normalize spectrogram
                spectrogram_data = np.log10(np.abs(spectrogram_data))

                # Plot the spectrogram in a new axis below the array plot
                ax_spec = axs[i + k]
                ax_spec.pcolormesh(adjusted_timestamps, f, spectrogram_data, shading='gouraud')
                # ax_spec.axis('off')
        i += 1
    axs[-1].set_xlabel('Time')

    # Display the figure
    if show_flag:
        plt.show()
    return fig, axs


def downsample_signal(signal_in, original_sampling_rate, target_sampling_rate):
    # Compute the decimation factor
    decimation_factor = int(original_sampling_rate / target_sampling_rate)

    # Design a low-pass filter with a cutoff frequency at the new Nyquist frequency (target_sampling_rate / 2)
    nyquist_freq = target_sampling_rate / 2.0
    b, a = butter(8, nyquist_freq / (original_sampling_rate / 2.0), btype='low')

    # Apply the low-pass filter
    filtered_signal = lfilter(b, a, signal_in)

    # Downsample the signal
    downsampled_signal = filtered_signal[::decimation_factor]

    return downsampled_signal


def remove_dc_offset(signal_in, sampling_rate, cutoff_freq=0.5):
    dc_offset = np.mean(signal_in)
    modified_signal = signal_in - dc_offset

    return modified_signal


def remove_mains_hum_v2(signal_data, sampling_rate):
    """
    Removes the mains hum from the input signal using the mean frequency subtraction method.

    Parameters:
        signal_data (numpy.ndarray): The input signal as a 1D NumPy array.
        sampling_rate (float): The sampling rate of the input signal (samples per second).

    Returns:
        numpy.ndarray: The modified signal with the mains hum removed.
    """
    # Step 1: Compute the FFT of the signal
    fft_result = np.fft.fft(signal_data)

    # Step 2: Calculate the mean value of each frequency component over time
    mean_frequency_values = np.mean(np.abs(fft_result), axis=0)

    # Step 3: Subtract the mean frequency values from the FFT matrix
    fft_result -= mean_frequency_values

    # Step 4: Convert back to the time domain using iFFT
    modified_signal = np.fft.ifft(fft_result).real

    return modified_signal


def remove_mains_hum(signal_data, sampling_rate, freq_to_remove=None):
    """
    Removes the mains hum from the input signal using a notch filter.

    Parameters:
        signal_data (numpy.ndarray): The input signal as a 1D NumPy array.
        sampling_rate (float): The sampling rate of the input signal (samples per second).

    Returns:
        numpy.ndarray: The modified signal with the mains hum removed.
    """
    # Calculate the FFT of the signal
    n = len(signal_data)
    fft_freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    # fft_vals = np.fft.fft(signal_data)
    if freq_to_remove is None:
        # Find the dominant frequency component close to 50 Hz
        target_freq = 50.0
        tolerance = 2.0  # Set a tolerance to account for small variations in frequency

        closest_idx = np.argmin(np.abs(fft_freqs - target_freq))
        dominant_freq = np.abs(fft_freqs[closest_idx])
        if np.abs(dominant_freq - target_freq) > tolerance:
            raise ValueError("Could not find the mains hum frequency component.")
    else:
        dominant_freq = freq_to_remove
    # Design the notch filter to remove the mains hum frequency
    Q = 10
    b, a = signal.iirnotch(dominant_freq, Q, sampling_rate)

    # Apply the notch filter to remove the mains hum
    modified_signal = signal.filtfilt(b, a, signal_data)

    return modified_signal


def get_signal_values_at_freq(signal_in, sample_rate, freq):
    # Compute the Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(signal_in)

    # Calculate the corresponding frequencies for each FFT bin
    fft_freqs = np.fft.fftfreq(len(signal_in), 1 / sample_rate)

    # Interpolate the FFT result to match the desired frequencies
    fft_interp = interp1d(fft_freqs, fft_result, kind='linear', bounds_error=False, fill_value=0.0)
    signal_values_at_freq = fft_interp(freq)

    return signal_values_at_freq


# # Example usage:
# if __name__ == "__main__":
#     # Generate some example data (replace this with your actual signal)
#     sampling_rate = 1000  # Replace this with the actual sampling rate of your signal
#     time = np.arange(0, 1, 1 / sampling_rate)
#     frequency = 10  # Frequency of the signal (in Hz)
#     signal = np.sin(2 * np.pi * frequency * time)
#
#     # Define the frequency range of interest (optional)
#     freq_range = (5, 15)  # Lower and upper frequency bounds (in Hz)
#
#     # Plot the entire spectrogram (no specific frequency range)
#     plot_spectrogram(signal, sampling_rate)
#
#     # Plot the spectrogram of the specific frequency range on a specific axes
#     fig, ax = plt.subplots()
#     plot_spectrogram(signal, sampling_rate, freq_range, axes=ax)


def pose_df_to_matrix(df, include_prob=False):
    dim_names = ['x', 'y', 'p']
    if not include_prob:
        dim_names = dim_names[:-1]
    n_dim = len(dim_names)
    n_joints = len(NODES2IND)
    n_df = len(df)
    # col_names = df.columns
    n_objects = 2
    ret = np.zeros((n_df, n_joints, n_dim, n_objects))
    for joint, joint_ind in NODES2IND.items():
        for ind_dim, dim_name in enumerate(dim_names):
            for obj in range(n_objects):
                col = f'{joint}_{obj + 1}_{dim_name}'
                vals = df[col].values
                ret[:, joint_ind, ind_dim, obj] = vals
    return ret


def fix_column_names(df):
    map_dict = {
        "Tail_end_1_x": "Tail_base_1_x",
        "Lat_right_1_x": "Lateral_right_1_x",
        "Tail_base_1_x": "Lateral_left_1_x",
        "Lat_left_1_x": "Trunk_1_x",
        "Center_1_x": "Neck_1_x"
    }
    new_dict = {}
    for k, v in map_dict.items():
        new_dict[k.replace('_1', '_2')] = v.replace('_1', '_2')

    map_dict.update(new_dict)
    for k, v in map_dict.items():
        new_dict[k.replace('_x', '_p')] = v.replace('_x', '_p')
    map_dict.update(new_dict)

    map_dict.update(new_dict)
    for k, v in map_dict.items():
        new_dict[k.replace('_x', '_y')] = v.replace('_x', '_y')
    map_dict.update(new_dict)

    return df.rename(columns=map_dict)


def fix_df_col_names_v2(df_to_fix):
    node_column_names_map = {
        # 'Ear_left': 'Ear_left',
        # 'Ear_right': 'Ear_right',
        # 'Nose': 'Nose',
        'Neck': 'Center',
        'Trunk': 'Lat_left',
        'Lateral_right': 'Lat_right',
        'Tail_base': 'Tail_end',
        'Lateral_left': 'Tail_base'
    }

    node_column_names_map_inv = {v: a for a, v in node_column_names_map.items()}
    old_col_names = df_to_fix.columns
    new_name_dict = {}
    # for new_name, old_name in node_column_names_map.items():

    for old_col_name in old_col_names:
        for new_name, old_name in node_column_names_map.items():
            if old_name in old_col_name:
                new_col_name = old_col_name.replace(old_name, new_name)
                new_name_dict[old_col_name] = new_col_name
                break
    return df_to_fix.rename(columns=new_name_dict)


def create_affine_transform_matrix(point1, point2):
    # Step 1: Translate point1 to the origin
    translation_matrix_1 = np.array([[1, 0, -point1[0]],
                                     [0, 1, -point1[1]],
                                     [0, 0, 1]])

    # Calculate the new coordinates of point2 after translation
    translated_point2 = np.dot(translation_matrix_1, np.array([point2[0], point2[1], 1]))

    # Step 2: Scale so that the distance between the points is 1
    distance = np.linalg.norm(translated_point2[:2])
    #     print(distance)

    scale_matrix = np.array([[1 / distance, 0, 0],
                             [0, 1 / distance, 0],
                             [0, 0, 1]])
    # Step 3: Rotate to align the line connecting the points with the vertical axis
    angle = np.arctan2(translated_point2[1], translated_point2[0]) + np.pi / 2

    #     print(angle* 180 / np.pi)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    #     print(rotation_matrix)
    # Step 4: Translate point2 to (0, 0.5)
    translation_matrix_2 = np.array([[1, 0, 0],
                                     [0, 1, 0.5],
                                     [0, 0, 1]])

    # Combine all transformations
    final_transform_matrix = np.dot(translation_matrix_2,
                                    np.dot(scale_matrix,
                                           np.dot(np.linalg.inv(rotation_matrix),
                                                  translation_matrix_1)))

    return final_transform_matrix


def apply_affine_transform(points, transformation_matrix):
    # points = points.reshape(-1,2)
    # Convert points to homogeneous coordinates (add a column of 1's)
    homogeneous_points = np.column_stack([points, np.ones(len(points))])
    # Apply the transformation matrix
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

    # Convert back to Cartesian coordinates (remove the last column)
    transformed_points = transformed_points[:, :-1]

    return transformed_points


def apply_reverse_affine_transform(transformed_points, transformation_matrix):
    # Convert transformed points to homogeneous coordinates (add a column of 1's)
    homogeneous_transformed_points = np.column_stack([transformed_points, np.ones(len(transformed_points))])

    # Calculate the inverse of the transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Apply the inverse transformation to get back to the original points
    original_points = np.dot(homogeneous_transformed_points, inverse_transformation_matrix.T)

    # Convert back to Cartesian coordinates (remove the last column)
    original_points = original_points[:, :-1]

    return original_points


def normalize_pose(pose_matrix, anchor_inds=None, anchors=("Center", "Tail_base"), inverse=False):
    if anchor_inds is None:
        anchor_inds = [NODES2IND[a] for a in anchors]

    dim = 2  # x-y
    points_1 = pose_matrix[:, anchor_inds[0], :dim, 0]
    points_2 = pose_matrix[:, anchor_inds[1], :dim, 0]

    n_elements, n_joints, n_dim, n_obj = pose_matrix.shape
    transformed_pose_matrix = np.zeros_like(pose_matrix)
    transformation_mats = np.zeros((n_elements,3,3))
    for ind in range(n_elements):
        transformation_mat = create_affine_transform_matrix(points_1[ind, :], points_2[ind, :])
        vals = pose_matrix[ind, :, :, :]
        vals = np.moveaxis(vals, 1, 2).reshape(n_joints * n_obj, n_dim)
        transformed_mat = apply_affine_transform(vals, transformation_mat)
        transformed_mat = transformed_mat.reshape(n_joints, n_obj, n_dim)
        transformed_mat = np.moveaxis(transformed_mat, 2, 1)
        transformed_pose_matrix[ind, :, :, :] = transformed_mat
        transformation_mats[ind, :, :] = transformation_mat
    return transformed_pose_matrix, transformation_mats


def apply_savgol_filter(df, window_length=10, polyorder=3):
    window_length = 10
    polyorder = 3
    return df.apply(lambda x: savgol_filter(x, window_length=window_length, polyorder=polyorder))


def plot_side_by_side(joints1, joints2, labels=None, match_axis=False, add_joint_names=False, edges1=None, edges2=None):
    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    if labels is None:
        labels = ['Set1', 'Set2']
    # Plot the first set of joints and edges on the first subplot
    axs[0].scatter(*zip(*joints1), c='b', label='Joints 1')

    if edges1 is None:
        edges1 = EDGES

    if edges2 is None:
        edges2 = EDGES

    for edge in edges1:
        axs[0].plot(*zip(*[joints1[i] for i in edge]), c='b', linestyle='--', alpha=0.7)

    if add_joint_names:
        for edge_name in NODES:
            ind_edge = NODES2IND[edge_name]
            x_text = joints1[ind_edge, 0]
            y_text = joints1[ind_edge, 1]
            str_text = edge_name

            axs[0].text(x_text, y_text, str_text)

    axs[0].set_title(labels[0])
    axs[0].legend()

    # Plot the second set of joints and edges on the second subplot
    axs[1].scatter(*zip(*joints2), c='r', label='Joints 2')
    for edge in edges2:
        axs[1].plot(*zip(*[joints2[i] for i in edge]), c='r', linestyle='--', alpha=0.7)
    axs[1].set_title(labels[1])
    axs[1].legend()

    #     # Set a common axis limit for both subplots
    if match_axis:
        minx, miny = np.min(np.vstack([joints1, joints2]), axis=0) - 1
        maxx, maxy = np.max(np.vstack([joints1, joints2]), axis=0) + 1
        axs[0].set_xlim(minx, maxx)
        axs[0].set_ylim(miny, maxy)
        axs[1].set_xlim(minx, maxx)
        axs[1].set_ylim(miny, maxy)

    if add_joint_names:
        for edge_name in NODES:
            ind_edge = NODES2IND[edge_name]
            x_text = joints1[ind_edge, 0]
            y_text = joints1[ind_edge, 1]
            str_text = edge_name

            axs[0].text(x_text, y_text, str_text)
    if add_joint_names:
        for edge_name in NODES:
            ind_edge = NODES2IND[edge_name]
            x_text = joints2[ind_edge, 0]
            y_text = joints2[ind_edge, 1]
            str_text = edge_name

            axs[1].text(x_text, y_text, str_text)

    # Display the plots
    plt.show()

from scipy.signal import butter, lfilter, spectrogram, filtfilt
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import pywt

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

def plot_spectrogram(signal_in: object, sampling_rate: object, freq_range: object = None, window: object = 'hann', nperseg: object = 256, noverlap: object = None,
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

    ax.pcolormesh(t, f, 10 * np.log10(np.abs(Sxx_filtered)+0.0001), shading='gouraud')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Spectrogram')

    if axes is None:
        plt.colorbar(label='Power Spectral Density (dB/Hz)')
        plt.show()


def plot_interpolated_arrays(arrays, timestamp_array, names=None, spectogram_dict_list=None, show_flag=True, fig=None, axs=None):
    # Create a figure with subplots
    if show_flag:
        fig, axs = plt.subplots(len(arrays)+len(spectogram_dict_list), 1, sharex='all', gridspec_kw={'hspace': 0.5})

    if not names is None:
        assert len(names) == len(arrays), 'number of names must match number of arrays'
    else:
        names = [f'Array {i}' for i in range(len(arrays)+len(spectogram_dict_list))]

    # Iterate over the arrays and plot in the respective subplots
    i = 0
    k = 0
    for array, name in zip(arrays, names):
        length = array.shape[1]
        adjusted_timestamps = np.linspace(timestamp_array[0, 0], timestamp_array[0, -1], length)
        axs[i+k].plot(adjusted_timestamps, array[0, :])
        axs[i+k].set_ylabel(name)

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
                f, t, spectrogram_data = plot_spectrogram(array[0], sampling_rate, freq_range=freq_range, window='hann', nperseg=nperseg, noverlap=noverlap,
                                 axes=None,
                                 show_flag=False)
                adjusted_timestamps = np.linspace(timestamp_array[0, 0], timestamp_array[0, -1], len(t))


                # spectrogram_data = calculate_spectrogram(array, timestamp_array, freq_range,
                #                                          spec_dict.get('spec_params', None))
                # normalize spectrogram
                spectrogram_data = np.log10(np.abs(spectrogram_data))

                # Plot the spectrogram in a new axis below the array plot
                ax_spec = axs[i+k]
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
    fft_freqs = np.fft.fftfreq(n, d=1/sampling_rate)
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
# Example usage:
if __name__ == "__main__":
    # Generate some example data (replace this with your actual signal)
    sampling_rate = 1000  # Replace this with the actual sampling rate of your signal
    time = np.arange(0, 1, 1 / sampling_rate)
    frequency = 10  # Frequency of the signal (in Hz)
    signal = np.sin(2 * np.pi * frequency * time)

    # Define the frequency range of interest (optional)
    freq_range = (5, 15)  # Lower and upper frequency bounds (in Hz)

    # Plot the entire spectrogram (no specific frequency range)
    plot_spectrogram(signal, sampling_rate)

    # Plot the spectrogram of the specific frequency range on a specific axes
    fig, ax = plt.subplots()
    plot_spectrogram(signal, sampling_rate, freq_range, axes=ax)

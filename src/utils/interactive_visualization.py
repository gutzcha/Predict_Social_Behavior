import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from utils.visualizations import load_example
from utils.helper_functions import plot_spectrogram, get_signal_values_at_freq, continuous_wavelet_transform
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os, sys
import matplotlib
# matplotlib.use('TkAgg')
from main.session_class import SocialExpSession
from coherence_gui import PlotCoherenceGUI


class SelectDialog(tk.Toplevel):
    def __init__(self, master, options, chosen):
        super().__init__(master)
        self.title("Select Modality")
        self.geometry("300x200")

        self.options = options
        self.chosen = chosen
        self.selected_options = []

        self.create_widgets()

    def create_widgets(self):
        self.checkboxes = []
        for option in self.options:
            var = tk.IntVar(value=1 if option in self.chosen else 0)
            checkbox = ttk.Checkbutton(self, text=option, variable=var)
            checkbox.pack(anchor="w")
            self.checkboxes.append((option, var))

        update_button = ttk.Button(self, text="Update", command=self.on_update)
        update_button.pack(pady=10)

    def on_update(self):
        self.chosen.clear()
        for option, var in self.checkboxes:
            if var.get() == 1:
                self.chosen.append(option)
        self.master.update_chosen_modalities(self.chosen)
        self.master.focus_set()
        self.destroy()


class SelectAreaDialog(tk.Toplevel):
    def __init__(self, master, options_dict):
        super().__init__(master)
        self.title("Select Modality")
        self.geometry("300x200")

        self.options_dict = options_dict

        self.create_widgets()

    def create_widgets(self):
        self.checkboxes = []
        for name, chosen in self.options_dict.items():
            var = tk.BooleanVar(value=chosen)
            checkbox = ttk.Checkbutton(self, text=name, variable=var)
            checkbox.pack(anchor="w")
            self.checkboxes.append((name, var))

        update_button = ttk.Button(self, text="Update", command=self.on_update)
        update_button.pack(pady=10)

    def on_update(self):
        for option, var in self.checkboxes:
            self.options_dict[option] = var.get()
        self.master.focus_set()
        self.destroy()


class SelectModalityDialog(tk.Toplevel):
    def __init__(self, master, options_choices):
        """

        :param master:
        :param options_choices: [{'modality',name, 'plot_vector', bool, 'plot_spectrogram', bool}]
        """
        super().__init__(master)
        self.title("Select Modality")
        self.geometry("300x200")

        self.options_choices = options_choices
        self.selected_options = []
        self.create_widgets()

    def create_widgets(self):
        self.checkboxes = []
        for ind, modality in enumerate(self.options_choices):
            name = modality['modality']

            val_array = tk.BooleanVar(value=modality['plot_vector'])
            val_spect = tk.BooleanVar(value=modality['plot_spectrogram'])

            # debug
            # print('creat widgets')
            # print(f'name:{name}, vector:{val_array.get()} spect:{val_spect.get()}')

            checkbox_array = tk.Checkbutton(self, text=name + ' array', variable=val_array, onvalue=True,
                                            offvalue=False)
            checkbox_array.grid(row=ind, column=0, padx=5, pady=5)

            checkbox_spect = tk.Checkbutton(self, text=name + ' spectrogram', variable=val_spect, onvalue=True,
                                            offvalue=False)
            checkbox_spect.grid(row=ind, column=1, padx=5, pady=5)

            self.checkboxes.append((name, val_array, val_spect))

        update_button = ttk.Button(self, text="Update", command=self.on_update)
        update_button.grid(row=len(self.options_choices), column=0, padx=5, pady=5, columnspan=2)

    def on_update(self):
        name_ind_map = {m['modality']: ind for ind, m in enumerate(self.options_choices)}
        for name, val_array, val_spect in self.checkboxes:
            # debug
            # print(f'name:{name}, val_array:{val_array.get()}, val_spect:{val_spect.get()}')
            if val_array.get():
                self.options_choices[name_ind_map[name]]['plot_vector'] = True
            else:
                self.options_choices[name_ind_map[name]]['plot_vector'] = False
            if val_spect.get():
                self.options_choices[name_ind_map[name]]['plot_spectrogram'] = True
            else:
                self.options_choices[name_ind_map[name]]['plot_spectrogram'] = False

        self.master.update_chosen_modalities()
        self.destroy()


class PlotApp(tk.Tk):
    def __init__(self, s_session: SocialExpSession):
        super().__init__()
        self.s_session = s_session
        self.time_range_start = 300
        self.time_range_end = 304
        self.step_size = 1

        # Set coherence gui handle
        self.coherence_handle = None

        self.modality_options = self.s_session.modality_options
        self.recorded_area_names = self.s_session.modalities['lfp'].recorded_area_names

        #  ['lfp', 'multispike', 'audio', 'usv']

        # self.area_to_load_default = ['EA', 'Mfb', 'MeD']
        self.area_to_load_default = self.recorded_area_names[:1]

        self.area_to_load = {a: a in self.area_to_load_default for a in self.recorded_area_names}

        # self.modality_plot_dict_list = [
        #     {'modality': 'lfp', 'array': [], 'plot_vector':True, 'plot_spectrogram': True, 'spec_dict':
        #         {'frequency_range': (1, 40), 'nperseg': 128, 'noverlap': 120,
        #          'sampling_rate': s_session.modalities['lfp'].sampling_rate}},
        #     {'modality': 'multispike', 'array': [], 'plot_vector': True, 'plot_spectrogram': False, 'spec_dict': {}},
        #     {'modality': 'audio', 'array': [], 'plot_vector': True, 'plot_spectrogram': True, 'spec_dict':
        #         {'frequency_range': (20000, 80000), 'nperseg': 256, 'noverlap': 200,
        #                                'sampling_rate': s_session.modalities['audio'].sampling_rate}},
        #     {'modality': 'usv', 'array': [], 'plot_vector': True, 'plot_spectrogram': False, 'spec_dict': {}},
        # ]

        # debug
        self.modality_plot_dict_list = [
            {'modality': 'lfp', 'array': [], 'plot_vector': False, 'plot_spectrogram': True, 'spec_dict':
                {'frequency_range': (1, 50), 'nperseg': 512, 'noverlap': 510,
                 'sampling_rate': s_session.modalities['lfp'].sampling_rate}},

            # {'modality': 'lfp_coherence', 'array': [], 'plot_vector': True, 'plot_spectrogram': False, 'areas':[('EA','MeD')]},


            {'modality': 'multispike', 'array': [], 'plot_vector': False, 'plot_spectrogram': False, 'spec_dict': {}},


            {'modality': 'audio', 'array': [], 'plot_vector': True, 'plot_spectrogram': False, 'spec_dict':
                {'frequency_range': (2000, 80000), 'nperseg': 1024, 'noverlap': np.floor(1024*0.7),
                 'sampling_rate': s_session.modalities['audio'].sampling_rate}},


            {'modality': 'usv', 'array': [], 'plot_vector': True, 'plot_spectrogram': False, 'spec_dict': {}},
        ]
        # array_dict: [{modality: 'lfp', array: [vector],  plot_vector: True, plot_spectrogram: True, spec_dict: None}]

        self.figure = None
        self.ax = None
        # _, self.ax = self.get_empty_plot()

        self.title("Plot App")
        # self.geometry("800x600")
        self.geometry("1200x600")

        self.create_widgets()

        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # Top container frame for buttons and text boxes
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X)


        # Entry for step size
        step_label = ttk.Button(top_frame, text="Update Step Size:", command=self.update_step_size)
        step_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.step_entry = ttk.Entry(top_frame, width=5)
        self.step_entry.insert(tk.END, str(self.step_size))
        self.step_entry.pack(side=tk.LEFT, padx=5, pady=5)

        # Forward and Backward buttons
        backward_button = ttk.Button(top_frame, text="Step Backward", command=self.backward)
        backward_button.pack(side=tk.LEFT, padx=5, pady=5)
        forward_button = ttk.Button(top_frame, text="Step Forward", command=self.forward)
        forward_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add Jump Forward and Jump Backward buttons
        jump_backward_button = ttk.Button(top_frame, text="Jump Backward", command=self.jump_backward)
        jump_backward_button.pack(side=tk.LEFT, padx=5, pady=5)
        jump_forward_button = ttk.Button(top_frame, text="Jump Forward", command=self.jump_forward)
        jump_forward_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add two text boxes to set range start and range end
        range_start_label = ttk.Label(top_frame, text="Range Start:")
        range_start_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.range_start_entry = ttk.Entry(top_frame, width=5)
        self.range_start_entry.insert(tk.END, str(self.time_range_start))
        self.range_start_entry.pack(side=tk.LEFT, padx=5, pady=5)

        range_end_label = ttk.Label(top_frame, text="Range End:")
        range_end_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.range_end_entry = ttk.Entry(top_frame, width=5)
        self.range_end_entry.insert(tk.END, str(self.time_range_end))
        self.range_end_entry.pack(side=tk.LEFT, padx=5, pady=5)

        # Add "Update Plot" button
        update_plot_button = ttk.Button(top_frame, text="Update Plot", command=self.on_update_plot)
        update_plot_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add choose modality button
        choose_modality_button = ttk.Button(top_frame, text="Choose Modality", command=self.open_modality_dialog)
        choose_modality_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add choose area button
        choose_area_button = ttk.Button(top_frame, text="Choose Areas", command=self.open_area_dialog)
        choose_area_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add choose coherence pair button

        self.coherence_checkbox_var = tk.BooleanVar(value=False)
        # self.coherence_checkbox_var.trace_add("write", self.open_coherence_gui)
        self.choose_coherence_button = tk.Checkbutton(top_frame, text="Choose coherence pair(s)", variable=self.coherence_checkbox_var,
                                                  command=self.open_coherence_gui)
        self.choose_coherence_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add stimulus insertion and removal lines state button
        # Not sure what to add here, maybe jump to insertion? or just display the time?

        # Bottom frame for the plot area
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        self.bottom_frame = bottom_frame

        # Plot area
        self.figure, self.ax = self.get_empty_plot()
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=bottom_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Navigation Toolbar
        self.toolbar_frame = ttk.Frame(bottom_frame)
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.plot_canvas, self.toolbar_frame)
        self.toolbar.update()

        self.update_plot()

    def _get_modalities_to_load(self):
        array_dict = self.modality_plot_dict_list
        modalities_to_load = []
        for m in array_dict:
            if m['plot_vector'] or m['plot_spectrogram']:
                modalities_to_load.append(m['modality'])
        return modalities_to_load

    def _get_n_subplots(self):
        n_subplots = 0
        array_dict = self.modality_plot_dict_list
        for m in array_dict:
            if m['plot_vector']:

                if m['modality'] in ['lfp', 'multispike']:
                    added = sum(np.array(list(self.area_to_load.values())) * 1)
                else:
                    added = 1
                n_subplots += added

            if m['plot_spectrogram']:

                if m['modality'] in ['lfp', 'multispike']:
                    added = sum(np.array(list(self.area_to_load.values())) * 1)
                else:
                    added = 1
                n_subplots += added

        return n_subplots


    def open_coherence_gui(self):
        if self.coherence_checkbox_var.get():
            self.coherence_handle = PlotCoherenceGUI(self)
        else:
            self.coherence_handle.on_close()
            self.coherence_handle = None





    def on_window_resize(self, event):
        # Resize the figure and canvas when the window is resized
        self.plot_canvas.get_tk_widget().configure(width=event.width, height=event.height)

    def _set_text_box(self, handle, value):
        handle.delete(0, tk.END)
        handle.insert(tk.END, str(value))

    def open_modality_dialog(self):
        dialog = SelectModalityDialog(self, self.modality_plot_dict_list)
        self.wait_window(dialog)

    def open_area_dialog(self):
        dialog = SelectAreaDialog(self, self.area_to_load)
        self.wait_window(dialog)
        self.update_plot()

    def update_chosen_modalities(self):

        if len(self.ax) != self._get_n_subplots():
            self.figure, self.ax = self.get_empty_plot()

        self.update_plot()
        print("Chosen Modalities:", self._get_modalities_to_load())

    def update_step_size(self):
        new_step_size = float(self.step_entry.get())
        if new_step_size > 0 and new_step_size <= self.s_session.get_max_range() / 4:
            self.step_size = new_step_size
        else:
            self._set_text_box(self.step_entry, self.step_size)

    def jump_forward(self):
        usv_data = self.s_session.modalities['usv_data'].data_dict['audio_usv_data'][['Begin_Time', 'End_Time']].values
        usv_data = usv_data[usv_data[:, 0] >= self.time_range_end, :]
        if usv_data.shape[0] == 0:
            return

        usv_data = usv_data[0, :]

        # set new range
        time_range_start = self.time_range_start
        time_range_end = self.time_range_end

        range_of_view = time_range_end - time_range_start
        range_of_new = usv_data[1] - usv_data[0]

        if range_of_view > range_of_new:
            # center
            diff_view = (range_of_view - range_of_new) / 2
        else:
            diff_view = 0

        new_start_range = np.max([usv_data[0] - diff_view, 0])
        new_end_range = np.min([usv_data[1] + diff_view, self.s_session.get_max_range()])

        self.range_start_entry.delete(0, tk.END)
        self.range_start_entry.insert(tk.END, str(new_start_range))

        self.range_end_entry.delete(0, tk.END)
        self.range_end_entry.insert(tk.END, str(new_end_range))

        self.on_update_plot()

    def jump_backward(self):

        usv_data = self.s_session.modalities['usv_data'].data_dict['audio_usv_data'][['Begin_Time', 'End_Time']].values
        usv_data = usv_data[usv_data[:, 1] <= self.time_range_start, :]
        if usv_data.shape[0] == 0:
            return

        usv_data = usv_data[-1, :]

        # set new range
        time_range_start = self.time_range_start
        time_range_end = self.time_range_end

        range_of_view = time_range_end - time_range_start
        range_of_new = usv_data[1] - usv_data[0]

        if range_of_view > range_of_new:
            # center
            diff_view = (range_of_view - range_of_new) / 2
        else:
            diff_view = 0

        new_start_range = np.max([usv_data[0] - diff_view, 0])
        new_end_range = np.min([usv_data[1] + diff_view, self.s_session.get_max_range()])

        self.range_start_entry.delete(0, tk.END)
        self.range_start_entry.insert(tk.END, str(new_start_range))

        self.range_end_entry.delete(0, tk.END)
        self.range_end_entry.insert(tk.END, str(new_end_range))

        self.on_update_plot()

    def get_empty_plot(self):
        n_subplots = self._get_n_subplots()
        # Create an empty plot and return figure and axis handles
        figure = self.figure
        if self.figure is None:
            figure, axs = plt.subplots(n_subplots, 1, sharex='all', gridspec_kw={'hspace': 0.5})
        else:
            axs = self.figure.subplots(n_subplots, 1, sharex='all', gridspec_kw={'hspace': 0.5})
        return figure, axs

    def on_close(self):
        # Clean up or perform any necessary actions before closing the app

        self.plot_canvas.get_tk_widget().destroy()  # Destroy the FigureCanvasTkAgg widget
        plt.close('all')  # Close all matplotlib figures
        self.quit()  # Quit the main loop
        self.destroy()  # Destroy the main window

    def forward(self):

        self.time_range_end = np.min([self.time_range_end + self.step_size, self.s_session.get_max_range()])

        if self.time_range_start + self.step_size < self.time_range_end:
            self.time_range_start += self.step_size
        else:
            self.time_range_start = np.max([self.time_range_end - self.step_size, 0])

        self.range_start_entry.delete(0, tk.END)
        self.range_start_entry.insert(tk.END, str(self.time_range_start))

        self.range_end_entry.delete(0, tk.END)
        self.range_end_entry.insert(tk.END, str(self.time_range_end))

        self.on_update_plot()

    def backward(self):

        self.time_range_start = np.max([self.time_range_start - self.step_size, 0])

        if self.time_range_end - self.step_size > self.time_range_start:
            self.time_range_end -= self.step_size
        else:
            self.time_range_end = np.min([self.time_range_start + self.step_size, self.s_session.get_max_range()])

        self.range_start_entry.delete(0, tk.END)
        self.range_start_entry.insert(tk.END, str(self.time_range_start))

        self.range_end_entry.delete(0, tk.END)
        self.range_end_entry.insert(tk.END, str(self.time_range_end))

        self.on_update_plot()

    def get_arrays(self, time_range=None):
        # Load data with updated time_range
        if time_range is None:
            time_range = (self.time_range_start, self.time_range_end)

        modalities_to_load = self._get_modalities_to_load()
        area_to_load_dict = self.area_to_load
        area_to_load = [a for a, v in area_to_load_dict.items() if v]
        timestamp_array, dict_to_return = self.s_session.load_data_in_range(
            time_range=time_range,
            area_name=area_to_load, modalities_to_load=modalities_to_load)

        return timestamp_array, dict_to_return

    def update_modality_plot_dict_list(self, array_dict, inplace=True):
        if inplace:
            modality_plot_dict_list = self.modality_plot_dict_list
        else:
            modality_plot_dict_list = [d.copy() for d in self.modality_plot_dict_list]
        for m in modality_plot_dict_list:
            if m['modality'] in array_dict.keys():
                m['array'] = array_dict[m['modality']]
        return modality_plot_dict_list

    def update_plot(self):
        timestamp_array, vector_list = self.get_arrays()
        _, vector_list_bg = self.get_arrays(time_range=(100, 150))

        array_dict = self.update_modality_plot_dict_list(vector_list)
        # bg_array_dict = self.update_modality_plot_dict_list(vector_list_bg, inplace=False)

        # Create the plot with updated data
        # Clear the previous subplots
        if self.ax is not None:
            for ax in self.ax:
                ax.clear()
            self.figure.clear()

        self.ax = self.get_empty_plot()[1]

        stimulus_insertion_removal_times = [self.s_session.stimulus_insertion_time,
                                            self.s_session.stimulus_removal_time]

        vertical_lines = [s for s in stimulus_insertion_removal_times if
                          self.time_range_start <= s <= self.time_range_end]

        self.plot_interpolated_arrays(timestamp_array, array_dict, vertical_lines=vertical_lines, subtract_areas=None)
        self.plot_canvas.draw()

        if self.coherence_handle is not None:
            self.coherence_handle.update_subplots()


    def on_update_plot(self):
        # Get the entered range values
        try:
            range_start = float(self.range_start_entry.get())
            range_end = float(self.range_end_entry.get())
        except ValueError:
            # If the entered values are not valid floats, do nothing

            return

        # Validate the range values
        max_range = self.s_session.get_max_range()
        if range_start >= range_end or range_start < 0 or range_end > max_range:
            # If the entered values are invalid, do nothing
            return

        # Update the time range and refresh the plot
        self.time_range_start = range_start
        self.time_range_end = range_end
        self.update_plot()

    def plot_interpolated_arrays(self, timestamp_array, array_dict=None,
                                 vertical_lines=None, bg_array_dict=None, subtract_areas=None, freq_to_remove=5):
        # Create a figure with subplots
        # array_dict: [{modality: 'lfp', array: [vector],  plot_vector: True, plot_spectrogram: True, spec_dict: None}]
        axs = self.ax
        n_subplots = self._get_n_subplots()
        assert n_subplots == len(axs), 'Number of axs must match the number of plots'

        # Iterate over the arrays and plot in the respective subplots
        i = 0

        for ind, m in enumerate(array_dict):
            array = m['array']
            name = m['modality']
            if m['plot_vector']:

                if len(array) == 0:
                    continue

                if name in ['lfp', 'multispike']:

                    for area, array_area in array.items():

                        if len(array_area) == 0:
                            continue
                        length = array_area.shape[1]

                        adjusted_timestamps = np.linspace(timestamp_array[0, 0], timestamp_array[0, -1], length)
                        axs[i].plot(adjusted_timestamps, array_area[0, :])

                        axs[i].set_ylabel(name + '_' + area)
                        i += 1
                else:
                    length = array.shape[1]
                    adjusted_timestamps = np.linspace(timestamp_array[0, 0], timestamp_array[0, -1], length)
                    axs[i].plot(adjusted_timestamps, array[0, :])
                    axs[i].set_ylabel(name)
                    i += 1

            if m['plot_spectrogram']:

                spec_dict = m.get('spec_dict', {})
                freq_range = spec_dict.get('frequency_range', None)
                nperseg = spec_dict.get('nperseg', 256)
                noverlap = spec_dict.get('noverlap', 200)
                sampling_rate = spec_dict.get('sampling_rate',
                                              1.0 / (timestamp_array[0, 1] - timestamp_array[0, 0]))
                if m['modality'] in ['lfp', 'multispike']:
                    # if not subtract_areas is None:
                    #
                    #     vec_sub = array[subtract_areas][0]
                    #     # vec_sub = remove_mains_hum(vec_sub, sampling_rate=self.s_session.modalities['lfp'].sampling_rate,
                    #     #                            freq_to_remove=freq_to_remove)
                    #     _, _, spectrogram_data_sub, _ = plot_spectrogram(vec_sub, sampling_rate, freq_range=freq_range,
                    #                                                      window='hann', nperseg=nperseg,
                    #                                                      noverlap=noverlap,
                    #                                                      axes=None,
                    #                                                      show_flag=False)
                    #     spectrogram_data_sub = np.log10(np.abs(spectrogram_data_sub))
                    for area, array_area in array.items():

                        if array_area.shape[1] == 0:
                            continue
                        # Calculate spectrogram using the provided data and parameters
                        # Use 'spec_params' key to access additional parameters required for spectrogram
                        vec = array_area[0]
                        # if not freq_to_remove is not None:
                        #     vec = remove_mains_hum(vec, sampling_rate=self.s_session.modalities['lfp'].sampling_rate,
                        #                            freq_to_remove=5)


                        # f, t, spectrogram_data, f_orig = plot_spectrogram(vec, sampling_rate, freq_range=freq_range,
                        #                                                   window='hann', nperseg=nperseg,
                        #                                                   noverlap=noverlap,
                        #                                                   axes=None,
                        #                                                   show_flag=False)

                        t, f, spectrogram_data = continuous_wavelet_transform(data=vec, fs=sampling_rate, f0=freq_range[0], fn=freq_range[1],
                                                                                     num_frequencies=100, wavelet='morl')
                        if not bg_array_dict is None:
                            adjusted_psd = get_signal_values_at_freq(bg_array_dict[ind]['array'][area][0],
                                                                     sampling_rate, f_orig)
                            freq_mask = (f_orig >= freq_range[0]) & (f_orig <= freq_range[1])
                            adjusted_psd = adjusted_psd[freq_mask]
                            adjusted_psd = adjusted_psd
                            spectrogram_data = spectrogram_data - adjusted_psd.reshape(-1, 1)
                            # spectrogram_data = spectrogram_data - np.min(spectrogram_data) + 0.0001
                        adjusted_timestamps = np.linspace(timestamp_array[0, 0], timestamp_array[0, -1], len(t))
                        spectrogram_data = np.log10(np.abs(spectrogram_data))
                        if not subtract_areas is None:
                            spectrogram_data -= spectrogram_data_sub
                        # normalize spectrogram



                        # Plot the spectrogram in a new axis below the array plot
                        ax_spec = axs[i]
                        p = ax_spec.pcolormesh(adjusted_timestamps, f, spectrogram_data, shading='gouraud')

                        # p.set_clim(-1,1.04)
                        # ax_spec.axis('off')
                        axs[i].set_ylabel(f'{name}_{area}_spec')
                        i += 1
                else:

                    # Calculate spectrogram using the provided data and parameters
                    # Use 'spec_params' key to access additional parameters required for spectrogram
                    f, t, spectrogram_data, _ = plot_spectrogram(array[0], sampling_rate, freq_range=freq_range,
                                                                 window='hann', nperseg=nperseg, noverlap=noverlap,
                                                                 axes=None,
                                                                 show_flag=False)

                    adjusted_timestamps = np.linspace(timestamp_array[0, 0], timestamp_array[0, -1], len(t))

                    # normalize spectrogram
                    spectrogram_data = np.log10(np.abs(spectrogram_data))

                    # Plot the spectrogram in a new axis below the array plot
                    ax_spec = axs[i]
                    ax_spec.pcolormesh(adjusted_timestamps, f, spectrogram_data, shading='gouraud')
                    # ax_spec.axis('off')
                    axs[i].set_ylabel(name + ' spec')
                    i += 1
        axs[-1].set_xlabel('Time')

        if not vertical_lines is None and len(vertical_lines) > 0:
            if len(vertical_lines) == 1:
                vertical_lines = [vertical_lines]
            for x_value in vertical_lines:
                for ax in axs:
                    line = ax.axvline(x=x_value, ymin=0, ymax=1, color='red', linestyle='-')
                    line.set_clip_on(False)


if __name__ == "__main__":
    # Assuming s_session is already initialized with the required data
    s_session = load_example(ind=9)
    print(s_session)
    app = PlotApp(s_session)
    app.mainloop()
    sys.exit(0)

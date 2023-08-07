import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from utils.visualizations import load_data_in_range, load_example
from utils.helper_functions import plot_interpolated_arrays
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class PlotApp(tk.Tk):
    def __init__(self, s_session):
        super().__init__()
        self.s_session = s_session
        self.time_range_start = 0
        self.time_range_end = 10
        self.step_size = 1

        self.spectrogram_dict_list = [{'index': 0, 'frequency_range': (1, 50), 'nperseg': 256, 'noverlap': 200,
                                 'sampling_rate': s_session.lfp.sampling_rate},
                                {'index': 2, 'frequency_range': (20000, 80000), 'nperseg': 256, 'noverlap': 200,
                                 'sampling_rate': s_session.audio.sampling_rate}
                                ]

        timestamp_array, vector_list, vector_names = self.get_arrays()
        self.n_subplots = len(vector_list) + len(self.spectrogram_dict_list)


        self.title("Plot App")
        self.geometry("800x600")



        self.create_widgets()

        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self.on_close)




    def create_widgets(self):
        # Entry for step size
        self.step_entry = ttk.Entry(self, width=10)
        self.step_entry.insert(tk.END, str(self.step_size))
        self.step_entry.pack(pady=10)

        # Forward and Backward buttons
        forward_button = ttk.Button(self, text="Forward", command=self.forward)
        backward_button = ttk.Button(self, text="Backward", command=self.backward)
        forward_button.pack(pady=5)
        backward_button.pack(pady=5)

        # Add Jump Forward and Jump Backward buttons
        jump_forward_button = ttk.Button(self, text="Jump Forward", command=self.jump_forward)
        jump_backward_button = ttk.Button(self, text="Jump Backward", command=self.jump_backward)
        jump_forward_button.pack(pady=5)
        jump_backward_button.pack(pady=5)

        # Add two text boxes to set range start and range end
        self.range_start_entry = ttk.Entry(self, width=10)
        self.range_start_entry.insert(tk.END, str(self.time_range_start))
        self.range_start_entry.pack(pady=5)

        self.range_end_entry = ttk.Entry(self, width=10)
        self.range_end_entry.insert(tk.END, str(self.time_range_end))
        self.range_end_entry.pack(pady=5)

        # Add "Update Plot" button
        update_plot_button = ttk.Button(self, text="Update Plot", command=self.on_update_plot)
        update_plot_button.pack(pady=5)

        # Plot area
        self.figure, self.ax = self.get_empty_plot()
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Navigation Toolbar
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.plot_canvas, self.toolbar_frame)
        self.toolbar.update()

        self.update_plot()

    def jump_forward(self):
        usv_data = self.s_session.usv_data.data_dict['audio_usv_data'][['Begin_Time', 'End_Time']].values
        usv_data = usv_data[usv_data[:, 1] >= self.time_range_start, :]
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
            diff_view = (range_of_view-range_of_new)/2
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

        usv_data = self.s_session.usv_data.data_dict['audio_usv_data'][['Begin_Time', 'End_Time']].values
        usv_data = usv_data[usv_data[:, 1] <= self.time_range_start,:]
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
            diff_view = (range_of_view-range_of_new)/2
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
        # Create an empty plot and return figure and axis handles
        fig, axs = plt.subplots(self.n_subplots, 1, sharex='all', gridspec_kw={'hspace': 0.5}, figsize=(10, 6))
        return fig, axs

    def on_close(self):
        # Clean up or perform any necessary actions before closing the app
        self.destroy()
        plt.close('all')  # Close all matplotlib figures
        self.quit()  # Quit the main loop

    def forward(self):
        self.time_range_start += self.step_size
        self.time_range_end += self.step_size
        self.update_plot()

    def backward(self):
        self.time_range_start -= self.step_size
        self.time_range_end -= self.step_size
        self.update_plot()

    def get_arrays(self):
        # Load data with updated time_range
        timestamp_array, vector_list, vector_names = load_data_in_range(self.s_session,
                                                                        time_range=(
                                                                            self.time_range_start, self.time_range_end),
                                                                        area_name='EA')
        return timestamp_array, vector_list, vector_names

    def update_plot(self):
        timestamp_array, vector_list, vector_names = self.get_arrays()

        # Create the plot with updated data
        [ax.clear() for ax in self.ax]


        plot_interpolated_arrays(vector_list, timestamp_array, vector_names, spectogram_dict_list=self.spectrogram_dict_list,
                                 show_flag=False, fig=self.figure, axs=self.ax)
        self.plot_canvas.draw()

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

if __name__ == "__main__":
    # Assuming s_session is already initialized with the required data
    s_session = load_example(ind=4)
    app = PlotApp(s_session)
    app.mainloop()

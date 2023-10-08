import matplotlib.pyplot as plt
from tkinter import ttk
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from interactive_visualization import PlotApp
from utils.visualizations import load_example


class SelectAreaPairsDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__()
        self.title("Select Area pairs")
        self.geometry("300x200")

        self.options_dict = master.area_to_load
        self.loaded_area_pairs = master.loaded_area_pairs
        # self.options_dict['EA'] = True

        self.create_widgets()

    def create_widgets(self):
        self.checkboxes = []
        for name, chosen in self.options_dict.items():
            boolvar = tk.BooleanVar(value=chosen)
            checkbox = tk.Checkbutton(self, text=name, variable=boolvar, command=self.on_click)
            checkbox.pack(anchor="w")
            self.checkboxes.append((name, boolvar, checkbox))

        self.update_button = ttk.Button(self, text="Update", command=self.on_update, state=tk.DISABLED)
        self.update_button.pack(pady=10)

        close_button = ttk.Button(self, text="Close", command=lambda: self.on_close)
        close_button.pack(pady=5)

    def get_selected_count(self):
        return sum(var.get() for _, var, _ in self.checkboxes)

    def enable_all(self):
        for name, var, cbox in self.checkboxes:
            cbox.config(state=tk.NORMAL)
        self.update_button.config(state=tk.DISABLED)

    def reset_all(self):
        for name, var, cbox in self.checkboxes:
            var.set(False)
        self.enable_all()

    def disable_non_selected(self):
        for name, var, cbox in self.checkboxes:
            if var.get():
                cbox.config(state=tk.NORMAL)
            else:
                cbox.config(state=tk.DISABLED)
        self.update_button.config(state=tk.NORMAL)

    def on_click(self):
        selected_count = self.get_selected_count()

        if selected_count > 2:
            self.reset_all()

        elif selected_count < 2:
            self.enable_all()

        else:
            # self.enable_all()
            self.disable_non_selected()

    def on_update(self):
        for option, var, _ in self.checkboxes:
            self.options_dict[option] = var.get()
        self.on_close()

    def on_close(self):
        self.master.focus_set()
        self.destroy()


class SubComponent:
    def __init__(self, parent_frame, area_name):
        self.is_empty = False
        self.parent_frame = parent_frame
        self.area_name = area_name

        self.sub_frame = tk.Frame(parent_frame)
        self.sub_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)  # Allow sub-component to expand

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.plot_area = self.figure.add_subplot(111)  # Use full area for plotting

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.sub_frame)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Allow canvas to expand

        self.text_panel = tk.Label(self.sub_frame, text=area_name)
        self.text_panel.pack(side=tk.LEFT)

        # self.plot_button = tk.Button(self.sub_frame, text="Update Plot", command=self.plot)
        # self.plot_button.pack(side=tk.LEFT)

        self.close_button = tk.Button(self.sub_frame, text="Close", command=self.close)
        self.close_button.pack(side=tk.LEFT)

        # self.plot()

    def plot(self, data_matrix=None, freq_range = (0,30)):
        self.plot_area.clear()

        if data_matrix is None:
            return

        if isinstance(data_matrix, dict):
            t = data_matrix['time']
            f = data_matrix['freq']
            data = data_matrix['data']

            mask = (f >= freq_range[0]) & (f <= freq_range[1])
            f = f[mask]
            data = data[mask]

            self.plot_area.pcolormesh(t, f, data)
        elif isinstance(data_matrix, np.ndarray):
            self.plot_area.pcolormesh(data_matrix)

        self.plot_area.get_figure().tight_layout()
        self.canvas.draw()

    def close(self):
        self.sub_frame.grid_forget()
        self.sub_frame.destroy()
        self.parent_frame.destroy()

        self.is_empty = True


class PlotCoherenceGUI(tk.Toplevel):
    def __init__(self, plot_app):
        super().__init__()

        self.title("Main GUI")
        self.subplot_array = []
        self.main_app = plot_app
        self.available_areas = self.main_app.recorded_area_names
        self.plotted_pairs = {}
        self.loaded_area_pairs = []

        self.create_navigation_panel()
        self.grid_rowconfigure(1, weight=1)

        # Debug
        data_matrix = np.random.random((10, 10))  # Example data matrix

        # Selection dialog
        self.area_to_load = {a: False for a in list(self.available_areas)}
        #
        # self.add_sub_plot(name='asd1', row=1, weight=1)
        # self.add_sub_plot(name='asd2', row=2, weight=1)
        self.update_subplots()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def get_areas_to_load(self):
        # return [a for a, val in self.area_to_load.items() if val]
        return [s.area_name for s in self.subplot_array]

    def get_arrays(self, time_range=None):

        # Load data with updated time_range
        if time_range is None:
            time_range = (self.main_app.time_range_start, self.main_app.time_range_end)
        areas_to_load = self.get_areas_to_load()
        timestamp_array = []
        lfp_co = []
        if len(areas_to_load) > 0:

            timestamp_array, dict_to_return = self.main_app.s_session.load_data_in_range(
                time_range=time_range,
                area_name=areas_to_load, modalities_to_load=['lfp_coherence'])

            lfp_co = dict_to_return['lfp_coherence']


        return timestamp_array, lfp_co

    def clean_subplot_array(self):
        empties = []
        for ind, a in enumerate(self.subplot_array):
            if a.is_empty:
                empties.append(ind)
        if empties:
            empties.sort(reverse=True)
            for ind in empties:
                del self.subplot_array[ind]

    def update_plot(self):
        pass

    def update_areas(self):
        pass

    def add_area_pair(self):
        pass

    def remove_area_pair(self):
        pass

    def set_pairs_not_plotted(self):
        pass

    def on_close(self):
        self.destroy()

    def on_open_select_areas(self):
        dialog = SelectAreaPairsDialog(self)
        self.wait_window(dialog)
        new_areas = [a for a, flag in self.area_to_load.items() if flag]
        if len(new_areas) == 2:
            self.add_new_area_pair(new_areas)
            self.update_subplots(len(self.subplot_array)-1)

    def add_new_area_pair(self, name):
        self.clean_subplot_array()
        n_sub = len(self.subplot_array)+1
        self.add_sub_plot(name, n_sub, 1)

    def add_sub_plot(self, name, row, weight):
        panel_frame = tk.Frame(self)
        panel_frame.grid(row=row, column=0, sticky="nsew")

        sub_comp = SubComponent(panel_frame, name)
        self.subplot_array.append(sub_comp)
        # self.grid_rowconfigure(0, weight=weight)
        # self.grid_columnconfigure(0, weight=1)
    def reformat_grid_weight(self):
        for ind in range(0, len(self.subplot_array)):
            self.grid_rowconfigure(ind+1, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def update_subplots(self, ind=None):
        self.clean_subplot_array()

        _ , data_dict = self.get_arrays()
        if ind is None:
            for splot in self.subplot_array:
                area_pair = tuple(splot.area_name)
                splot.plot(data_dict[area_pair])
        else:
            splot = self.subplot_array[ind]
            area_pair = tuple(splot.area_name)
            splot.plot(data_dict[area_pair])
        self.reformat_grid_weight()

    def create_navigation_panel(self):
        # Navigation panel
        self.navigation_panel_frame = tk.Frame(self)
        self.navigation_panel_frame.grid(row=0, column=0, sticky="nsew")
        # self.navigation_panel_frame.grid_rowconfigure(0, weight=1)
        # self.navigation_panel_frame.grid_columnconfigure(0, weight=1)

        # Add buttons
        # choose pairs

        # Forward and Backward buttons
        select_areas_button = ttk.Button(self.navigation_panel_frame, text="Add pair",
                                         command=self.on_open_select_areas)
        select_areas_button.pack(side=tk.LEFT, padx=5, pady=5)

        # forward_button = ttk.Button(self.navigation_panel_frame, text="Add pair", command=self.forward)
        # forward_button.pack(side=tk.LEFT, padx=5, pady=5)

        # update plots

        #

# if __name__ == "__main__":
#     s_session = load_example(ind=9)
#     main_app = PlotApp(s_session)
#     app = PlotCoherenceGUI(main_app)
#     app.mainloop()


# Example usage
# if __name__ == "__main__":
#     main_gui = tk.Tk()
#     main_gui.title("Main GUI")
#
#     options = {"Option 1": False, "Option 2": False, "Option 3": False, "Option 4": False}
#     select_dialog = SelectAreaPairsDialog(main_gui, options)
#
#     main_gui.mainloop()

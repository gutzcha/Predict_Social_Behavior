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
            print('creat widgets')
            print(f'name:{name}, vector:{ val_array.get()} spect:{ val_spect.get()}')

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
            print(f'name:{name}, val_array:{val_array.get()}, val_spect:{val_spect.get()}')
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

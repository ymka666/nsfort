import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
import nsfort
from parsing import Parser
from time import time


url = {
    'oschadbank.ua': 'https://api.oschadbank.ua/rates/archive?type=0&currency=USD&filter=year&format=xlsx'
    }

column_names = {
    'oschadbank.ua': ['Купівля', 'Продаж', 'Курс Нбу']
    }

ctk.set_appearance_mode('dark')
ctk.set_default_color_theme("blue")


class Demonstration(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Demonstration")
        self.geometry("900x530+500+250")
        self.resizable(False, False)
        self.fonts = {'small': ctk.CTkFont(size=14, weight="bold"),
                      'large': ctk.CTkFont(size=22, weight="bold")}

        self.grid_columnconfigure((0, 1, 2), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.first_block_components()
        self.second_block_components()
        self.third_block_components()
        self.forth_block_components()

        self.raw_data = None
        self.current_data = None
        self.model_expo = 0
        self.model_transcendent = 0
        self.model_mlq = 0
        self.text_info = ""
        self.used_approximation = []
        self.extrapolation_coef = 0

    def define_default_values(self):
        self.raw_data = None
        self.current_data = None
        self.model_expo = 0
        self.model_transcendent = 0
        self.model_mlq = 0
        self.text_info = ""
        self.used_approximation = []
        self.extrapolation_coef = 0

    def paint_graph(self, default_data, data, points=False):
        plt.clf()
        plt.plot(default_data, label='default_data')
        for name, values in data.items():
            plt.plot(values, label=name)

        if points:
            for point in points:
                plt.plot(point[0], point[1], 'ro')
        plt.legend()
        plt.show()

    def try_value(self, value, integer=False):
        try:
            value = float(value)
            return int(value) if integer else value
        except ValueError:
            return None

    def paint_hist(self, data):
        plt.clf()
        plt.hist(data, bins=20, facecolor="blue", alpha=0.5)
        plt.show()

    def get_made_model(self):
        size = self.try_value(self.n_entry.get(), integer=True) or 10000
        coef = self.try_value(self.coef_entry.get()) or 0.0000005
        mean = self.try_value(self.mean_entry.get()) or 0
        spread = self.try_value(self.spread_entry.get()) or 5
        anomaly_percent = self.try_value(
            self.anomaly_percent_entry.get()) or 10
        anomaly_error = self.try_value(self.anomaly_error_entry.get()) or 3
        n_anomaly = int((size * anomaly_percent) / 100)

        made_model = nsfort.ideal_quadratic_model(size, coef)
        self.text_info += 'Модель виміру(квадратичний закон), нормальний шум'
        if self.anomaly_checkbox.get():
            made_model = nsfort.noise_anomaly_model(made_model, mean, spread,
                                                    n_anomaly, anomaly_error)
            self.text_info += ' та аномальні вимірювання'
        else:
            made_model = nsfort.noise_model(made_model, mean, spread)
        self.text_info += '\n'
        self.paint_hist(made_model)
        return made_model

    def get_current_data(self):
        self.text_info += 'Обрані дані:\n\n'
        chosen_data = self.radio_var_data.get()
        if chosen_data in [0, 1, 2]:
            data_name = 'oschadbank.ua'
            parser = Parser(url[data_name])
            data_column = column_names[data_name][chosen_data]
            current_data = parser.get_data(data_column)
            self.text_info += f'{data_name} UAH до USD, {data_column}\n'
        else:
            current_data = self.get_made_model()
        self.get_characteristics(current_data)
        return current_data

    def get_sliding_data(self, data):
        sliding_coef = self.try_value(self.sliding_coef_entry.get(),
                                      integer=True) or 5
        sliding_data = nsfort.sliding_window(data, sliding_coef)
        self.paint_graph(self.raw_data, {'sliding_window': sliding_data})
        self.text_info += '\nХарактеристики даних, очещених від АВ:\n'
        self.get_characteristics(sliding_data)
        return sliding_data

    def define_extrapolation_coef(self):
        try:
            extrapolation_coef = float(self.extrapolation_coef_entry.get())
        except ValueError:
            extrapolation_coef = 0.5
        return extrapolation_coef

    def cut_data(self, data):
        cut_coef = self.try_value(self.cut_coef_entry.get()) or 0
        coef_left = (100 - cut_coef) / 100
        cut_data = data[:int(coef_left * len(data))]
        if cut_coef:
            self.text_info +=\
                f'\nХарактеристики даних, обрізаних на {cut_coef}%:\n'
            self.get_characteristics(cut_data)
        return cut_data

    def get_characteristics(self, data, t_time=0):
        characteristics = nsfort.statistical_characteristics(data)
        self.text_info +=\
            f'Математичне сподівання: {characteristics["mean"]}\n'
        self.text_info += f'Дисперсія: {characteristics["variance"]}\n'
        self.text_info += f'СКВ: {characteristics["standard_deviation"]}\n'
        if t_time:
            self.text_info += f'Час обрахунку: {t_time}\n'

    def get_global_linear_deviation(self, data_vector, data_approximation):
        global_deviation = nsfort.global_linear_deviation(data_vector,
                                                          data_approximation)
        self.text_info += f'Динамічна похибка моделі: {global_deviation}\n'

    def define_exponent_approximation(self, data):
        if self.exponent_checkbox.get():
            self.model_expo = nsfort.ExponentialModel(data)
            start = time()
            data, _ = self.model_expo.approximate(coef=self.extrapolation_coef)
            result_time = time() - start

            self.text_info += '\nЕкспоненційна модель:\n' + self.model_expo.view() + '\n'
            self.get_characteristics(data, result_time)
            self.get_global_linear_deviation(self.model_expo.data_vector, data)

    def define_transcendent_approximation(self, data):
        if self.transcendent_checkbox.get():
            self.model_transcendent = nsfort.TranscendentModel(data)
            start = time()
            data, _ = self.model_transcendent.approximate(coef=self.extrapolation_coef)
            result_time = time() - start

            self.text_info += '\nТрансцендентна модель:\n' + self.model_transcendent.view() + '\n'
            self.get_characteristics(data, result_time)
            self.get_global_linear_deviation(self.model_transcendent.data_vector, data)

    def define_mlq_approximation(self, data):
        if self.mlq_checkbox.get():
            try:
                linear_power = int(self.mlq_power_entry.get())
            except ValueError:
                linear_power = 2

            self.model_mlq = nsfort.MLQModel(data, power=linear_power)
            start = time()
            data, _ = self.model_mlq.approximate(coef=self.extrapolation_coef)
            result_time = time() - start

            self.text_info += '\nMLQ:\n' + self.model_mlq.view() + '\n'
            self.get_characteristics(data, result_time)
            self.get_global_linear_deviation(self.model_mlq.data_vector, data)

    def get_point(self, model):
        size = model.size
        end = model.output[size][0]
        return [size, end]

    def set_all_parameters(self):
        self.define_default_values()
        self.current_data = self.get_current_data()
        self.raw_data = self.current_data

        if self.radio_var_sliding.get():
            self.current_data = self.get_sliding_data(self.current_data)

        self.extrapolation_coef = self.define_extrapolation_coef()

        self.text_info += '\nФункції апроксимації:\n'
        data = self.cut_data(self.current_data)
        self.define_exponent_approximation(data)
        self.define_transcendent_approximation(data)
        self.define_mlq_approximation(data)

        points = []
        data_paint = {}

        if self.model_expo:
            data_paint['Exponential'] = self.model_expo.output
            if self.extrapolation_coef:
                points.append(self.get_point(self.model_expo))

        if self.model_transcendent:
            data_paint['Transcendent'] = self.model_transcendent.output
            if self.extrapolation_coef:
                points.append(self.get_point(self.model_transcendent))

        if self.model_mlq:
            data_paint['MLQ'] = self.model_mlq.output
            if self.extrapolation_coef:
                points.append(self.get_point(self.model_mlq))

        if data_paint:
            self.paint_graph(self.current_data, data_paint, points)
            self.open_info_window()

    def open_info_window(self):
        info_window = ctk.CTkToplevel()
        info_window.title("Information")
        info_window.resizable(False, False)
        textbox = ctk.CTkTextbox(info_window, width=750, height=600)
        textbox.grid(row=0, column=0, padx=(20, 20),
                     pady=(20, 20), sticky="nsew")

        textbox.insert("0.0", self.text_info)

    def first_block_components(self):
        """Front компоненти 1 блоку"""
        data_frame = ctk.CTkFrame(self)
        data_frame.grid(row=0, column=0, padx=(20, 10),
                        pady=(10, 10), sticky="nsew")
        self.radio_var_data = tk.IntVar(value=0)

        data_label = ctk.CTkLabel(data_frame, text="Вибір даних",
                                  font=self.fonts['large'])
        data_label.grid(row=1, column=0, padx=50, pady=(20, 10))

        real_data_label = ctk.CTkLabel(data_frame,
                                       text="Реальні дані\n Ощадбанк (1 рік)",
                                       font=self.fonts['small'])
        real_data_label.grid(row=2, column=0, columnspan=1,
                             padx=10, pady=10, sticky="")
        self.buy_data_rb = ctk.CTkRadioButton(data_frame, text='Купівля',
                                              variable=self.radio_var_data,
                                              value=0)
        self.buy_data_rb.grid(row=3, column=0, pady=10, padx=20, sticky="n")
        self.sell_data_rb = ctk.CTkRadioButton(data_frame, text='Продаж',
                                               variable=self.radio_var_data,
                                               value=1)
        self.sell_data_rb.grid(row=4, column=0, pady=10, padx=20, sticky="n")
        self.nbu_data_rb = ctk.CTkRadioButton(data_frame, text='Курс Нбу',
                                              variable=self.radio_var_data,
                                              value=2)
        self.nbu_data_rb.grid(row=5, column=0, pady=10, padx=20, sticky="n")

        model_data_label = ctk.CTkLabel(data_frame, text="Штучні дані",
                                        font=self.fonts['small'])
        model_data_label.grid(row=6, column=0, columnspan=1,
                              padx=10, pady=10, sticky="")
        self.quadratic_rb = ctk.CTkRadioButton(data_frame, text='Квадратичні',
                                               variable=self.radio_var_data,
                                               value=3)
        self.quadratic_rb.grid(row=7, column=0, pady=10, padx=20, sticky="n")
        tabview = ctk.CTkTabview(data_frame, width=20)
        tabview.grid(row=8, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        tabview.add("Розмір")
        tabview.add("Шум")
        tabview.add("АВ")

        self.n_entry = ctk.CTkEntry(tabview.tab("Розмір"),
                                    placeholder_text="n=10000")
        self.n_entry.grid(row=0, column=0, padx=(40, 0),
                          pady=(5, 5), sticky="n")
        self.coef_entry = ctk.CTkEntry(tabview.tab("Розмір"),
                                       placeholder_text="coef=0.0000005")
        self.coef_entry.grid(row=1, column=0, padx=(40, 0),
                             pady=(5, 5), sticky="n")

        self.mean_entry = ctk.CTkEntry(tabview.tab("Шум"),
                                       placeholder_text="mean=0")
        self.mean_entry.grid(row=0, column=0, padx=(40, 0),
                             pady=(5, 5), sticky="n")
        self.spread_entry = ctk.CTkEntry(tabview.tab("Шум"),
                                         placeholder_text="spread=5")
        self.spread_entry.grid(row=1, column=0, padx=(40, 0),
                               pady=(5, 5), sticky="n")

        checkbox_text = 'Наявність\nаномальних\nвимірювань'
        self.anomaly_checkbox = ctk.CTkCheckBox(tabview.tab("АВ"),
                                                text=checkbox_text)
        self.anomaly_checkbox.grid(row=0, column=0, padx=(40, 0),
                                   pady=(5, 5), sticky="n")
        self.anomaly_percent_entry = ctk.CTkEntry(tabview.tab("АВ"),
                                                  placeholder_text="% = 10")
        self.anomaly_percent_entry.grid(row=1, column=0, padx=(40, 0),
                                        pady=(5, 5), sticky="n")
        self.anomaly_error_entry = ctk.CTkEntry(tabview.tab("АВ"),
                                                placeholder_text="error = 3")
        self.anomaly_error_entry.grid(row=2, column=0, padx=(40, 0),
                                      pady=(5, 5), sticky="n")

    def second_block_components(self):
        """Front компоненти 2 блоку"""
        sliding_frame = ctk.CTkFrame(self)
        sliding_frame.grid(row=0, column=1, padx=(10, 10),
                           pady=(10, 10), sticky="nsew")
        self.radio_var_sliding = tk.IntVar(value=0)

        anomaly_label = ctk.CTkLabel(sliding_frame,
                                     text="Очищення\nвід АВ",
                                     font=self.fonts['large'])
        anomaly_label.grid(row=1, column=1, padx=30, pady=(20, 10))
        sliding_on_rb = ctk.CTkRadioButton(sliding_frame, text='Ввімкнути',
                                           variable=self.radio_var_sliding,
                                           value=1)
        sliding_on_rb.grid(row=2, column=1, pady=10, padx=20, sticky="n")
        sliding_off_rb = ctk.CTkRadioButton(sliding_frame, text='Вимкнути',
                                            variable=self.radio_var_sliding,
                                            value=0)
        sliding_off_rb.grid(row=3, column=1, pady=10, padx=20, sticky="n")
        sliding_coef_label = ctk.CTkLabel(sliding_frame, text="Коефіцієнт",
                                          font=self.fonts['small'])
        sliding_coef_label.grid(row=4, column=1, columnspan=1,
                                padx=10, pady=10, sticky="")
        self.sliding_coef_entry = ctk.CTkEntry(sliding_frame,
                                               placeholder_text="5", width=25)
        self.sliding_coef_entry.grid(row=5, column=1, padx=(30, 30),
                                     pady=(5, 5), sticky="nsew")

    def third_block_components(self):
        """Front компоненти 3 блоку"""
        func_frame = ctk.CTkFrame(self)
        func_frame.grid(row=0, column=2, padx=(10, 10),
                        pady=(10, 10), sticky="nsew")

        func_label = ctk.CTkLabel(func_frame, text="Вибір функції",
                                  font=self.fonts['large'])
        func_label.grid(row=1, column=0, padx=24, pady=(20, 10))
        non_linear_label = ctk.CTkLabel(func_frame, text="Нелінійна",
                                        font=self.fonts['small'])
        non_linear_label.grid(row=2, column=0, columnspan=1,
                              padx=10, pady=10, sticky="")

        self.exponent_checkbox = ctk.CTkCheckBox(func_frame, text='exponential')
        self.exponent_checkbox.grid(row=3, column=0, padx=20,
                                    pady=10, sticky="n")
        self.exponent_checkbox.select()
        self.transcendent_checkbox = ctk.CTkCheckBox(func_frame, text='transcendent')
        self.transcendent_checkbox.grid(row=4, column=0, padx=20,
                                        pady=10, sticky="n")

        linear_label = ctk.CTkLabel(func_frame, text="Лінійна",
                                    font=self.fonts['small'])
        linear_label.grid(row=5, column=0, columnspan=1,
                          padx=10, pady=10, sticky="")
        self.mlq_checkbox = ctk.CTkCheckBox(func_frame,
                                            text='квадратичний\nМНК')
        self.mlq_checkbox.grid(row=6, column=0, padx=(18, 0),
                               pady=10, sticky="n")
        self.mlq_power_entry = ctk.CTkEntry(func_frame,
                                            placeholder_text="2", width=10)
        self.mlq_power_entry.grid(row=7, column=0, padx=(40, 20),
                                  pady=(5, 5), sticky="nsew")

    def forth_block_components(self):
        """Front компоненти 4 блоку"""
        extrapolation_frame = ctk.CTkFrame(self)
        extrapolation_frame.grid(row=0, column=3, padx=(10, 20),
                                 pady=(10, 10), sticky="nsew")

        extrapolation_label = ctk.CTkLabel(extrapolation_frame,
                                           text="Апроксимація",
                                           font=self.fonts['large'])
        extrapolation_label.grid(row=1, column=0, padx=20, pady=(20, 20))
        extrapolation_coef_label = ctk.CTkLabel(extrapolation_frame,
                                                text="Коефіцієнт\nекстраполяції",
                                                font=self.fonts['small'])
        extrapolation_coef_label.grid(row=2, column=0, padx=20, pady=(20, 5))
        self.extrapolation_coef_entry = ctk.CTkEntry(extrapolation_frame,
                                                     placeholder_text="0.5")
        self.extrapolation_coef_entry.grid(row=3, column=0, padx=(30, 30),
                                           pady=(5, 5), sticky="nsew")
        cut_coef_label = ctk.CTkLabel(extrapolation_frame,
                                      text="Обрізати % даних\n(опціонально)",
                                      font=self.fonts['small'])
        cut_coef_label.grid(row=4, column=0, padx=20, pady=(20, 5))
        self.cut_coef_entry = ctk.CTkEntry(extrapolation_frame,
                                           placeholder_text="0")
        self.cut_coef_entry.grid(row=5, column=0, padx=(30, 30),
                                 pady=(5, 5), sticky="nsew")
        graph_button = ctk.CTkButton(extrapolation_frame, text='Візуалізація',
                                     text_color=("gray10", "#DCE4EE"),
                                     command=self.set_all_parameters)
        graph_button.grid(row=6, column=0, padx=(20, 20),
                          pady=(20, 20), sticky="nsew")


if __name__ == '__main__':
    app = Demonstration()
    app.mainloop()

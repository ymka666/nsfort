import pandas as pd
import numpy as np
import wget


class Parser:
    def __init__(self, api: str) -> None:
        self.api = api
        self.file_name = self.data_download(self.api)

    def data_download(self, api: str) -> str:
        return wget.download(api)

    def get_data(self, column_name: str) -> np.ndarray:
        file = pd.read_excel(self.file_name)
        values = file[column_name].tolist()

        data = np.zeros((len(values)))
        for i, value in enumerate(values):
            data[i] = value

        return data

    def get_name(self) -> str:
        return self.file_name

import math as mt
import numpy as np


class ApproximationModel:
    """
    Main class used for creation standart parameters and methods
    for approximation  model
    """
    def __init__(self, data: list[float]) -> None:
        """
        Parametrs
        ----------
        _data: list, np.ndarray
            data for approximation
        _size: int
            size of data for approximation
        _data_vector: np.ndarray
            vector of data for approximayion
        _extrapolation_coef: float, int
            extrapolation coefficient
        _extrapolation_num: int
            amount of data to extrapolate
        _mlq_coefs_vector: np.ndarray
            vector of MLQ coefficients
        _basis_matrix: np.ndarray
            matrix of basis coefficients
        _coefs_vector: np.ndarray
            vector of approximation coefficients
        _output: np.ndarray
            approximation data
        _output_size: int
            size of approximation data
        """
        self._data = data
        self._size = len(self._data)
        self._data_vector = self.__define_data_vector()
        self._extrapolation_coef = 0
        self._extrapolation_num = 0
        self._mlq_coefs_vector = None
        self._basis_matrix = None
        self._coefs_vector = None
        self._output = None
        self._output_size = 0

    def __define_data_vector(self) -> np.ndarray:
        """
        Creation of data vector

        Returns
        ----------
        data_vector: np.ndarray
            vector of current data
        """
        data_vector = np.zeros((self._size, 1))
        for row in range(self._size):
            data_vector[row, 0] = float(self._data[row])
        return data_vector

    def _define_basis_matrix(self, rows: int, columns: int) -> np.ndarray:
        """
        Creation of basis matrix

        Parametrs
        ----------
        rows: int
            number of rows
        columns: int
            number of columns

        Returns
        ----------
        basis_matrix: np.ndarray
            matrix of basis functions
        """
        basis_matrix = np.ones((rows, columns))
        for row in range(rows):
            for column in range(1, columns):
                basis_matrix[row, column] = float(row**column)
        return basis_matrix

    def _algorithm_mlq(self, data_vector: np.ndarray,
                       basis_matrix: np.ndarray) -> np.ndarray:
        """
        Method of least squares algorithm

        Parametrs
        ----------
        data_vector: np.ndarray
            matrix of basis functions
        basis_matrix: np.ndarray
            matrix of basis functions

        Returns
        ----------
        mlq_coefs_vector: np.ndarray

        """
        f_t = basis_matrix.T
        fft = f_t.dot(basis_matrix)
        ffti = np.linalg.inv(fft)
        fftift = ffti.dot(f_t)
        mlq_coefs_vector = fftift.dot(data_vector)
        return mlq_coefs_vector

    def _define_extrapolation_num(self, coef: float) -> None:
        """
        Defines extrapolation number

        Parametrs
        ----------
        coef: flaot
            extrapolation coefficient
        """
        self._extrapolation_coef = coef
        self._extrapolation_num = mt.ceil(
            self._size * self._extrapolation_coef
            )

    def _define_mlq_coefs_vector(self) -> None:
        """
        Defines vector of MLQ coefficients
        """
        self._mlq_coefs_vector = self._algorithm_mlq(self._data_vector,
                                                     self._basis_matrix)

    def _difine_output(self) -> None:
        """
        Define standart output data
        """
        self._output_size = self._size + self._extrapolation_num
        self._output = np.zeros((self._output_size, 1))

    @property
    def data(self) -> list[float]:
        return self._data

    @property
    def size(self) -> int:
        return self._size

    @property
    def data_vector(self) -> np.ndarray:
        return self._data_vector

    @property
    def extrapolation_coef(self) -> float:
        return self._extrapolation_coef

    @property
    def extrapolation_num(self) -> int:
        return self._extrapolation_num

    @property
    def mlq_coefs_vector(self) -> np.ndarray:
        return self._mlq_coefs_vector

    @property
    def basis_matrix(self) -> np.ndarray:
        return self._basis_matrix

    @property
    def coefs_vector(self) -> np.ndarray:
        return self._coefs_vector

    @property
    def output(self) -> np.ndarray:
        return self._output

    @property
    def output_size(self) -> int:
        return self._output_size


class ExponentialModel(ApproximationModel):
    """
    Class for approximating data with nonlinear exponential model
    """
    def __init__(self, data: list[float]) -> None:
        """
        Parametrs
        ----------
        data: list, np.ndarray
            data for approximation
        _basis_matrix: np.ndarray
            matrix of basis functions
        _coef_a3: float
            the fourth coefficient of the approximating function
        _coef_a2: flaot
            the third coefficient of the approximating function
        _coef_a1: flaot
            the second coefficient of the approximating function
        _coef_a0: flaot
            the first coefficient of the approximating function
        """
        super().__init__(data)
        self._basis_matrix = self._define_basis_matrix(self.size, 4)
        self._coef_a3 = 0
        self._coef_a2 = 0
        self._coef_a1 = 0
        self._coef_a0 = 0

    def __exponential_algorithm(self) -> None:
        """
        Algorithm for calculating the exponential approximating function
        """
        self._define_mlq_coefs_vector()

        self._coef_a3 = 3 * (self._mlq_coefs_vector[3, 0] / self._mlq_coefs_vector[2, 0])
        self._coef_a2 = (2 * self._mlq_coefs_vector[2, 0]) / (self._coef_a3 ** 2)
        self._coef_a1 = self._mlq_coefs_vector[1, 0] - (self._coef_a2 * self._coef_a3)
        self._coef_a0 = self._mlq_coefs_vector[0, 0] - self._coef_a2

    def approximate(self, coef: float = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Exponential approximation

        Parametrs
        ----------
        coef: flaot
            extrapolation coefficient (default = 0)

        Returns
        ----------
        output: np.ndarray
            approximation data
        coefs_vector: np.ndarray
            vector of approximation coefficients
        """
        if coef:
            self._define_extrapolation_num(coef)
        self.__exponential_algorithm()
        self._difine_output()

        for i in range(self._output_size):
            self._output[i, 0] = self._coef_a0 + self._coef_a1 * i + self._coef_a2 * mt.exp(self._coef_a3 * i)

        self._coefs_vector = np.ndarray(shape=(4, 1),
                                        buffer=np.array([self._coef_a0,
                                                         self._coef_a1,
                                                         self._coef_a2,
                                                         self._coef_a3])
                                        )

        return self.output, self.coefs_vector

    def view(self) -> str:
        """
        General view of the approximating function
        """
        return 'y(t) = {0} + {1} * t + {2} * exp({3} * t)'.format(*self._coefs_vector)

    def simple_view(self) -> str:
        """
        An example of the general form of the approximating function
        """
        return 'y(t) = a0 + a1 * t + a2 * exp(a3 * t)'

    @property
    def coef_a3(self) -> float:
        return self._coef_a3

    @property
    def coef_a2(self) -> float:
        return self._coef_a2

    @property
    def coef_a1(self) -> float:
        return self._coef_a1

    @property
    def coef_a0(self) -> float:
        return self._coef_a0


class TranscendentModel(ApproximationModel):
    """
    Class for approximating data with nonlinear transcendential model
    """
    def __init__(self, data: list[float]) -> None:
        """
        Parametrs
        ----------
        data: list, np.ndarray
            data for approximation
        _basis_matrix: np.ndarray
            matrix of basis functions
        _coef_a0: float
            coefficient of the approximating function
        _coef_w0: flaot
            coefficient of the approximating function
        _coef_b0: flaot
            cefficient of the approximating function
        """
        super().__init__(data)
        self._basis_matrix = self._define_basis_matrix(self.size, 3)
        self._coef_a0 = 0
        self._coef_w0 = 0
        self._coef_b0 = 0

    def __transcendent_algorithm(self) -> None:
        """
        Algorithm for calculating the transcendental approximating function
        """
        self._define_mlq_coefs_vector()

        self._coef_a0 = self._mlq_coefs_vector[0, 0]
        self._coef_w0 = mt.sqrt(abs(self._mlq_coefs_vector[2, 0] / self._mlq_coefs_vector[0, 0]))
        self._coef_b0 = self._mlq_coefs_vector[1, 0] / self._coef_w0

    def approximate(self, coef: float = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Transcendent approximation

        Parametrs
        ----------
        coef: flaot
            extrapolation coefficient (default = 0)

        Returns
        ----------
        output: np.ndarray
            approximation data
        coefs_vector: np.ndarray
            vector of approximation coefficients
        """
        if coef:
            self._define_extrapolation_num(coef)
        self.__transcendent_algorithm()
        self._difine_output()

        for i in range(self._output_size):
            self._output[i, 0] = (self._coef_a0 * mt.cos(self._coef_w0*i) +\
                                  self._coef_b0 * mt.sin(self._coef_w0*i))

        self._coefs_vector = np.ndarray(shape=(3, 1),
                                        buffer=np.array([self._coef_a0,
                                                         self._coef_b0,
                                                         self._coef_w0])
                                        )

        return self._output, self._coefs_vector

    def view(self) -> str:
        """
        General view of the approximating function
        """
        return 'y(t) = {0} * cos({2} * t) + {1} * sin({2} * t)'.format(*self._coefs_vector)

    def simple_view(self) -> str:
        """
        An example of the general form of the approximating function
        """
        return 'y(t) = a0 * cos(w0 * t) + b0 * sin(w0 * t)'

    @property
    def coef_a0(self) -> float:
        return self._coef_a0

    @property
    def coef_w0(self) -> float:
        return self._coef_w0

    @property
    def coef_b0(self) -> float:
        return self._coef_b0


class MLQModel(ApproximationModel):
    """
    Class for approximating data with Method of Least Squares
    """
    def __init__(self, data: list[float], power: int = 2) -> None:
        """
        Parametrs
        ----------
        data: list, np.ndarray
            data for approximation
        _basis_matrix: np.ndarray
            matrix of basis functions
        power: int
            the degree of the polynomial
        """
        super().__init__(data)
        self._power = power
        self._basis_matrix = self._define_basis_matrix(self._size,
                                                       self._power + 1)

    def approximate(self, coef: float = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Method of Least Squares approximation

        Parametrs
        ----------
        coef: flaot
            extrapolation coefficient (default = 0)

        Returns
        ----------
        output: np.ndarray
            approximation data
        coefs_vector: np.ndarray
            vector of approximation coefficients
        """
        if coef:
            self._define_extrapolation_num(coef)
        self._define_mlq_coefs_vector()
        self._coefs_vector = self._mlq_coefs_vector
        self._difine_output()

        for i in range(self._output_size):
            for j in range(self._power+1):
                self._output[i, 0] += self._coefs_vector[j, 0] * i**j

        return self._output, self._coefs_vector

    def view(self) -> str:
        """
        General view of the approximating function
        """
        view = f'y(t) = [{self._coefs_vector[0, 0]}] +'
        for i in range(1, self._power+1):
            view += f' [{self._coefs_vector[i, 0]}] * t^{i} +'
        view = view[:-1]
        return view

    def simple_view(self) -> str:
        """
        An example of the general form of the approximating function
        """
        view = 'y(t) = a0 +'
        for i in range(1, self._power+1):
            view += f' a{i} * t^{i} +'
        view = view[:-1]
        return view

    @property
    def power(self) -> int:
        return self._power


def sliding_window(data: list[float] | np.ndarray,
                   n_window: int) -> np.ndarray:
    """
    A method for cleaning the sample from anomalous measurements
    """
    size = len(data)
    j_window = mt.ceil(size-n_window) + 1
    data_window = np.zeros((n_window))
    mid_i = np.zeros(size)

    for j in range(j_window):
        for i in range(n_window):
            ind = j + i
            data_window[i] = data[ind]
        mid_i[ind] = np.median(data_window)

    result_data = np.zeros(size)
    for j in range(size):
        result_data[j] = mid_i[j]
    for j in range(n_window):
        result_data[j] = data[j]

    return result_data


def statistical_characteristics(data: np.ndarray) -> dict:
    """
    Calculation of mean, variance and standard deviation of sample
    """
    mean = np.median(data)
    variance = np.var(data)
    standard_deviation = mt.sqrt(variance)
    characteristics = {
        'mean': mean,
        'variance': variance,
        'standard_deviation': standard_deviation
    }

    return characteristics


def ideal_quadratic_model(size: int,
                          coef: float) -> np.ndarray:
    """
    Creating a perfect quadratic curve
    """
    arr = np.zeros(size)
    for i in range(size):
        arr[i] = coef * i**2

    return arr


def gaussian_distribution(mean: float,
                          spread: float,
                          size: int) -> np.ndarray:
    """
    Gaussian distribution
    """
    return np.random.normal(mean, spread, size)


def uniform_nums_distribution(low: int,
                              high: int,
                              size: int) -> np.ndarray:
    """
    Uniform distribution of int numbers
    """
    arr = np.zeros(size)
    for i in range(size):
        arr[i] = np.random.randint(low, high)

    return arr


def noise_model(model: np.ndarray,
                mean_noise: float,
                spread_noise: float) -> np.ndarray:
    """
    Creating an artificial model with normal noise

    Parametrs
    ----------
    model: np.ndarray
        ideal data for model creation
    mean_noise: float
        mean of noise that will be added
    spread_noise: float
        spread of noise that will be added

    Returns
    ----------
    data: np.ndarray
    """
    size = len(model)
    data = np.zeros(size)
    noise = gaussian_distribution(mean_noise, spread_noise, size)

    for i in range(size):
        data[i] = model[i] + noise[i]

    return data


def noise_anomaly_model(model: np.ndarray,
                        mean: float,
                        spread: float,
                        n_anomaly: int,
                        anomaly_error: float) -> np.ndarray:
    """
    Creating an artificial model with normal noise and abnormal measurements
    """
    result_model = noise_model(model, mean, spread)
    anomalies = gaussian_distribution(mean, (anomaly_error*spread), n_anomaly)

    size = len(result_model)
    anomaly_index = uniform_nums_distribution(0, size, size)

    for i in range(n_anomaly):
        k = int(anomaly_index[i])
        result_model[k] = model[k] + anomalies[i]

    return result_model


def global_linear_deviation(data_vector: np.ndarray,
                            data_approximation: np.ndarray) -> float:
    """
    global linear deviation of the estimate - dynamic error of the model
    """
    size = len(data_vector)
    delta = 0
    for i in range(size):
        delta = delta + abs(data_vector[i, 0] - data_approximation[i, 0])
    global_deviation = delta / (size + 1)

    return global_deviation

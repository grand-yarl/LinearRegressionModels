import numpy as np
import pandas as pd
import pywt


class LinearRegression:
    """
    Ordinary Linear Regression class
        attributes:
            Coeff - numpy array for regression coefficients list,

            Free_coeff - free regression coefficient,

            Square_error - statistic square error for model on train data,

            R2_error - R2 error coefficient for model on train data.
        setting variables:
            number_of_singulars - number of singular values used to find pseudo inverse matrix.
        methods:
            static_methods:
                prepare_data - preparing full data from dataframe to test and train samples.

            object_methods:
                fit_model - find model coefficients on train samples,

                predict - find prediction values for input parameters,

                test_error - calculate prediction error for test samples.
    """

    def __init__(self, set_number_of_singulars=-1):
        """
        Linear Regression class constructor
            inputs:
                number_of_singulars - number of singular values used to find pseudo inverse matrix.
        """
        self.Coeff = np.array([])
        self.Free_coeff = None
        self.Square_error = None
        self.R2_error = None

        self.number_of_singulars = set_number_of_singulars

    @staticmethod
    def prepare_data(dataframe, x_names_list, y_name, test_size=0.2):
        """
        Data preparation method
            inputs:
                dataframe - pandas dataframe with full data,

                x_names_list - names of model input parameters,

                y_name - name of model output parameter,

                test_size - size of test data sample (0.2 by default).
            outputs:
                x_train - train input sample,

                y_train - train output sample,

                x_test - test input sample,

                y_test - test output sample.
        """
        if (test_size > 1) or (test_size < 0):
            print("Test size must be from 0. to 1.")
            return None
        x = dataframe[x_names_list].to_numpy()
        y = dataframe[y_name].to_numpy()
        x_train, x_test, y_train, y_test = LinearRegression.__split_train_test__(x, y, test_size=test_size)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def __split_train_test__(x, y, test_size=0.33):
        """
        Method for splitting full data on train and test samples (private)
            inputs:
                x - full input data,

                y - full output data,

                test_size - size of test data sample (0.2 by default).
            outputs:
                x_train - train input sample,

                y_train - train output sample,

                x_test - test input sample,

                y_test - test output sample.
        """
        number_of_test = round(len(y) * test_size)
        number_of_train = len(y) - number_of_test
        x_train = x[0: number_of_train, :]
        x_test = x[number_of_train: len(y), :]
        y_train = y[0: number_of_train]
        y_test = y[number_of_train: len(y)]
        return x_train, x_test, y_train, y_test

    def fit_model(self, x, y):
        """
        Method for calculating model coefficients (Coeff and Free_coeff)
            inputs:
                x - train input sample,

                y - train output sample.
            outputs:
                Coeff - numpy array for regression coefficients list,

                Free_coeff - free regression coefficient.
        """
        ones = np.array([np.ones(len(x))])
        x_new = np.append(x, ones.T, axis=1)
        x_pseudo = self.__pseudo_inv__(x_new)
        self.Coeff = np.dot(x_pseudo, y.T)
        self.Free_coeff = self.Coeff[len(self.Coeff) - 1]
        self.Coeff = np.delete(self.Coeff, len(self.Coeff) - 1)
        self.Square_error, self.R2_error = self.test_error(x, y)
        return self.Coeff, self.Free_coeff

    def __pseudo_inv__(self, matrix):
        """
        Method for calculating pseudo inverse matrix (private)
            input:
                matrix - matrix, from which pseudo inverse is calculating.
            output:
                matrix_pseudo_inv - pseudo inverse matrix.
        """
        if self.number_of_singulars == -1:
            self.number_of_singulars = np.linalg.matrix_rank(matrix)
        u, s, vt = np.linalg.svd(matrix)
        s_inv = np.zeros(len(s))

        for i in range(self.number_of_singulars):
            if i > len(s) - 1:
                break
            if s[i] != 0:
                s_inv[i] = 1 / s[i]

        sigma = np.zeros(np.shape(matrix))
        sigma[:len(s), :len(s)] = np.diag(s_inv)
        matrix_pseudo_inv = np.dot(np.dot(vt.T, sigma.T), u.T)
        return matrix_pseudo_inv

    def predict(self, x_entry):
        """
        Method for predicting output values
            input:
                x_entry - input parameters that are used for prediction.
            output:
                y_predicted - list of predicted values.
        """
        if self.Free_coeff is None:
            print("Find coefficients for model first")
            return None
        else:
            y_predicted = np.dot(self.Coeff, x_entry.T) + self.Free_coeff
            return y_predicted

    def test_error(self, x_test, y_test):
        """
        Method for calculating errors on test sample
            input:
                x_test: test input sample,

                y_test: test output sample.
            output:
                square_error: square error on test sample,

                r2_error: r2 error coefficient for test sample.
        """
        if self.Free_coeff is None:
            print("Find coefficients for model first")
            return None
        else:
            if x_test.shape[0] != len(y_test):
                print("Test data X and Y must be the same shape")
                return None
            ss_res = 0
            ss_tot = 0
            y_predicted = self.predict(x_test)
            for i in range(len(y_test)):
                ss_res += (y_test[i] - y_predicted[i]) ** 2
                ss_tot += (y_test[i] - np.mean(y_test)) ** 2
            r2_error = 1 - ss_res / ss_tot
            square_error = np.sqrt(ss_res) / (len(y_test) - 2)
        return square_error, r2_error


class AssociativeLinearRegression(LinearRegression):
    """
    Associative Linear Regression class (Heir of LinearRegression class)
        description:
            For every prediction point create new model, that include associated train set,
            regression coefficients and free coefficient. The association criterion is minkowski
            distance between current input vector and associated vector.
        attributes:
            Coeff - list of regression coefficients for all point models,

            Free_coeff - list of free regression coefficients for all point models,

            Square_error - statistic square error for model on current test data,

            R2_error - R2 error coefficient for model on current test data,

            X_knowledge_base - set of input vectors, from which associative selection is performing,

            Y_knowledge_base - set of output values, that is connected with X_knowledge_base.
        setting variables:
            number_of_singulars - number of singular values used to find pseudo inverse matrix,

            minkowski_level - level for Minkowski distance. If equals 1 - Manhattan distance, 2 - Euclid distance (default),

            max_radius - maximum distance, which allowed for association (infinite for default),

            number_of_train_set - number of the nearest vectors, that can be associated (infinite for default).
        methods:
            object_methods:
                fit_model - find model coefficients on associated set,

                predict - find prediction values for input parameters,

                test_error - calculate prediction error for test samples,

                update_knowledge_base - Inserting the latest data to knowledge base.
    """
    def __init__(self, x_knowledge_base, y_knowledge_base, set_number_of_singulars=-1, set_minkowski_level=2,
                 set_max_radius=np.inf, set_number_of_train_set=np.inf):
        """
        Associative Linear Regression class constructor
            input
                x_knowledge_base - set of input vectors, from which associative selection is performing,

                y_knowledge_base - set of output values, that is connected with X_knowledge_base,

                set_number_of_singulars - number of singular values used to find pseudo inverse matrix,

                set_minkowski_level - level for Minkowski distance. If equals 1 - Manhattan distance, 2 - Euclid distance (default),

                set_max_radius - maximum distance, which allowed for association (infinite for default),

                set_number_of_train_set - number of the nearest vectors, that can be associated (infinite for default).
        """
        super().__init__(set_number_of_singulars)

        self.X_knowledge_base = x_knowledge_base
        self.Y_knowledge_base = y_knowledge_base

        self.minkowski_level = set_minkowski_level
        self.max_radius = set_max_radius
        self.number_of_train_set = set_number_of_train_set

    def fit_model(self, x, y):
        ones = np.array([np.ones(len(x))])
        x_new = np.append(x, ones.T, axis=1)
        x_pseudo = self.__pseudo_inv__(x_new)
        coeff = np.dot(x_pseudo, y.T)
        free_coeff = coeff[len(coeff) - 1]
        coeff = np.delete(coeff, len(coeff) - 1)
        return coeff, free_coeff

    def __associative_set__(self, current_x):
        """
        Method for collecting associated train set (private)
            input:
                current_x - current vector, for which associated set is constructing.
            outputs:
                x_associated - associated set of input vectors,

                y_associated - associated set of output values.
        """
        x_associated = np.array([])
        y_associated = np.array([])
        distance_associated = np.array([])

        for i in range(len(self.X_knowledge_base)):
            distance = 0

            for j in range(len(current_x)):
                distance += abs(current_x[j] - self.X_knowledge_base[i, j]) ** self.minkowski_level
            distance = distance ** (1 / self.minkowski_level)

            if distance == 0.0:
                continue

            if distance <= self.max_radius:
                if len(x_associated) == 0:
                    x_associated = np.copy(self.X_knowledge_base[i])
                else:
                    x_associated = np.vstack([x_associated, self.X_knowledge_base[i]])
                y_associated = np.append(y_associated, self.Y_knowledge_base[i])
                distance_associated = np.append(distance_associated, distance)

        if len(y_associated) <= 1:
            raise ArithmeticError("The restrictions are too strong, no associations found."
                                  " Try to make set_max_radius bigger")

        while self.number_of_train_set < len(y_associated):
            excess = np.argmax(distance_associated)
            x_associated = np.delete(x_associated, excess, axis=0)
            y_associated = np.delete(y_associated, excess)
            distance_associated = np.delete(distance_associated, excess)
        return x_associated, y_associated

    def predict(self, x_entry):
        if (len(self.X_knowledge_base) == 0) or (len(self.Y_knowledge_base) == 0):
            print("The knowledge base is empty. Add data to knowledge base")
            return None
        if len(self.X_knowledge_base) != len(self.Y_knowledge_base):
            print("Shapes of X and Y knowledge base must be equal")
            return None
        self.Coeff = np.zeros((x_entry.shape[0], x_entry.shape[1]))
        self.Free_coeff = np.zeros(x_entry.shape[0])
        y_predicted = np.array([])
        for i in range(x_entry.shape[0]):
            x_associative, y_associative = self.__associative_set__(x_entry[i])
            self.Coeff[i, :], self.Free_coeff[i] = self.fit_model(x_associative, y_associative)
            y_predicted = np.append(y_predicted, np.dot(self.Coeff[i, :], x_entry[i]) + self.Free_coeff[i])
        return y_predicted

    def test_error(self, x_test, y_test):
        if x_test.shape[0] != len(y_test):
            print("Test data X and Y must be the same shape")
            return None
        if (len(self.X_knowledge_base) == 0) or (len(self.Y_knowledge_base) == 0):
            print("The knowledge base is empty. Add data to knowledge base")
            return None
        if len(self.X_knowledge_base) != len(self.Y_knowledge_base):
            print("Shapes of X and Y knowledge base must be equal")
            return None
        ss_res = 0
        ss_tot = 0
        y_pred = self.predict(x_test)
        for i in range(len(y_test)):
            ss_res += (y_test[i] - y_pred[i]) ** 2
            ss_tot += (y_test[i] - np.mean(y_test)) ** 2
        self.Square_error = np.sqrt(ss_res) / (len(y_test) - 2)
        self.R2_error = 1 - ss_res / ss_tot
        return self.Square_error, self.R2_error

    def update_knowledge_bases(self, x_new, y_new):
        """
        Method for inserting the latest data to knowledge base
            inputs:
                x_new - new set of input vectors,

                y_new - new set of output values.
        """
        if x_new.shape[0] != len(y_new):
            print("Test data X and Y must be the same shape")
            return None
        for i in range(len(y_new)):
            self.X_knowledge_base = np.vstack([self.X_knowledge_base, x_new[i]])
            self.Y_knowledge_base = np.append(self.Y_knowledge_base, y_new[i])
        return


class WaveletAssociativeLinearRegression(AssociativeLinearRegression):

    def __init__(self, x_knowledge_base, y_knowledge_base, set_number_of_singulars=-1, set_minkowski_level=2,
                 set_max_radius=np.inf, set_number_of_train_set=np.inf, set_wavelet="haar", set_dwt_depth=1):
        """
        Wavelet Associative Linear Regression class constructor
            input
                x_knowledge_base - set of input vectors, from which associative selection is performing,

                y_knowledge_base - set of output values, that is connected with X_knowledge_base,

                set_number_of_singulars - number of singular values used to find pseudo inverse matrix,

                set_minkowski_level - level for Minkowski distance. If equals 1 - Manhattan distance, 2 - Euclid distance (default),

                set_max_radius - maximum distance, which allowed for association (infinite for default),

                set_number_of_train_set - number of the nearest vectors, that can be associated (infinite for default),

                set_wavelet - type of discrete wavelet, used in discrete wavelet decomposition (haar wavelet for default)

                set_dwt_depth - depth for discrete wavelet decomposition (1 for default)
        """
        if set_wavelet not in pywt.wavelist(kind='discrete'):
            raise AttributeError("Invalid wavelet type. Check by pywt.wavelist(kind='discrete')")

        if len(y_knowledge_base) < 2 ** set_dwt_depth:
            raise Warning("DWT depth is too big, it can cause coefficient error in decomposition")

        super().__init__(x_knowledge_base, y_knowledge_base, set_number_of_singulars, set_minkowski_level,
                         set_max_radius, set_number_of_train_set)

        self.wavelet = set_wavelet
        self.dwt_depth = set_dwt_depth

        self.Dwt_coeff_list = [[np.array([]) for k in range(np.shape(x_knowledge_base)[1])] for j in range(self.dwt_depth + 1)]
        for j in range(np.shape(x_knowledge_base)[1]):
            full_coeffs = pywt.wavedec(x_knowledge_base[:, j], self.wavelet, level=self.dwt_depth, mode='smooth')
            for i in range(self.dwt_depth + 1):
                if len(self.Dwt_coeff_list[i][j]) == 0:
                    self.Dwt_coeff_list[i][j] = np.copy(full_coeffs[i])
                else:
                    self.Dwt_coeff_list[i][j] = np.vstack([self.Dwt_coeff_list[i][j], full_coeffs[i]])


def __associative_set__(self, current_time, x_test):
    """
    Method for collecting  associated by wavelet transform train set (private)
        input:
            current_time - current time for vector, for which associated set is constructing.

            x_test - test
        outputs:
            x_associated - associated set of input vectors,

            y_associated - associated set of output values.
    """
    x_associated = np.array([])
    y_associated = np.array([])
    distance_associated = np.array([])

    current_dwt_coeff = [[0. for k in range(np.shape(x_test)[1])] for j in range(self.dwt_depth + 1)]
    for j in range(np.shape(x_test)[1]):
        full_coeffs = pywt.wavedec(x_test[:, j], self.wavelet, level=self.dwt_depth, mode='smooth')
        for i in range(self.dwt_depth + 1):
            if i == 0:
                c = current_time // 2 ** self.dwt_depth
            else:
                c = current_time // 2 ** (self.dwt_depth - i + 1)
            current_dwt_coeff[i][j] = full_coeffs[i][c]

    for i in range(len(self.X_knowledge_base)):
        distance = 0

        for m in range(self.dwt_depth + 1):
            for n in range(np.shape(x_test)[1]):
                if m == 0:
                    it = current_time // 2 ** self.dwt_depth
                else:
                    it = i // 2 ** (self.dwt_depth - m + 1)
                distance += (current_dwt_coeff[m][n] - self.Dwt_coeff_list[m][n][it]) ** self.minkowski_level
            distance = distance ** (1 / self.minkowski_level)

        if distance == 0.0:
            continue

        if distance <= self.max_radius:
            if len(x_associated) == 0:
                x_associated = np.copy(self.X_knowledge_base[i])
            else:
                x_associated = np.vstack([x_associated, self.X_knowledge_base[i]])
            y_associated = np.append(y_associated, self.Y_knowledge_base[i])
            distance_associated = np.append(distance_associated, distance)

    if len(y_associated) <= 1:
        raise ArithmeticError("The restrictions are too strong, no associations found."
                              " Try to make set_max_radius bigger")

    while self.number_of_train_set < len(y_associated):
        excess = np.argmax(distance_associated)
        x_associated = np.delete(x_associated, excess, axis=0)
        y_associated = np.delete(y_associated, excess)
        distance_associated = np.delete(distance_associated, excess)
    return x_associated, y_associated


def predict(self, x_entry):
    if (len(self.X_knowledge_base) == 0) or (len(self.Y_knowledge_base) == 0):
        print("The knowledge base is empty. Add data to knowledge base")
        return None
    if len(self.X_knowledge_base) != len(self.Y_knowledge_base):
        print("Shapes of X and Y knowledge base must be equal")
        return None
    self.Coeff = np.zeros((x_entry.shape[0], x_entry.shape[1]))
    self.Free_coeff = np.zeros(x_entry.shape[0])
    y_predicted = np.array([])
    for i in range(x_entry.shape[0]):
        x_associative, y_associative = self.__associative_set__(i, x_entry)
        self.Coeff[i, :], self.Free_coeff[i] = self.fit_model(x_associative, y_associative)
        y_predicted = np.append(y_predicted, np.dot(self.Coeff[i, :], x_entry[i]) + self.Free_coeff[i])
    return y_predicted

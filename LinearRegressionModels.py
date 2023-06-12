import numpy as np
import pandas as pd


class LinearRegression:

    def __init__(self, number_of_singulars_=-1):
        self.Coeff = np.array([])
        self.B = None
        self.Square_error = None
        self.R2_error = None
        self.number_of_singulars = number_of_singulars_

    @staticmethod
    def prepare_data(dataframe, x_names_list, y_name, test_size=0.2):
        x = dataframe[x_names_list].to_numpy()
        y = dataframe[y_name].to_numpy()
        x_train, x_test, y_train, y_test = LinearRegression.__split_train_test__(x, y, test_size=test_size)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def __split_train_test__(x, y, test_size=0.33):
        number_of_test = round(len(y) * test_size)
        number_of_train = len(y) - number_of_test
        x_train = x[0: number_of_train, :]
        x_test = x[number_of_train: len(y), :]
        y_train = y[0: number_of_train]
        y_test = y[number_of_train: len(y)]
        return x_train, x_test, y_train, y_test

    def fit_model(self, x, y):
        ones = np.array([np.ones(len(x))])
        x_new = np.append(x, ones.T, axis=1)
        x_pseudo = self.__pseudo_inv__(x_new)
        self.Coeff = np.dot(x_pseudo, y.T)
        self.B = self.Coeff[len(self.Coeff) - 1]
        self.Coeff = np.delete(self.Coeff, len(self.Coeff) - 1)
        self.Square_error, self.R2_error = self.test_error(x, y)
        return

    def __pseudo_inv__(self, matrix):
        if self.number_of_singulars == -1:
            self.number_of_singulars = np.linalg.matrix_rank(matrix)
        u, s, vt = np.linalg.svd(matrix)
        s_inv = s
        # зануление малых сингулярных чисел
        for i in range(len(s)):
            if (i < self.number_of_singulars) and (s[i] != 0):
                s_inv[len(s) - i - 1] = 1 / s[len(s) - i - 1]
            else:
                s_inv[len(s) - i - 1] = 0
        sigma = np.zeros(np.shape(matrix))
        sigma[:min(np.shape(matrix)[0], np.shape(matrix)[1]), :min(np.shape(matrix)[0], np.shape(matrix)[1])] = np.diag(s_inv)
        matrix_pseudo_inv = np.dot(np.dot(vt.T, sigma.T), u.T)
        return matrix_pseudo_inv

    def test_error(self, x_test, y_test):
        if self.B is None:
            print("Find coefficients for model first")
            return None
        else:
            if x_test.shape[0] != len(y_test):
                print("Test data X and Y must be the same shape")
                return None
            ss_res = 0
            ss_tot = 0
            y_pred = self.predict(x_test)
            for i in range(len(y_test)):
                ss_res += (y_test[i] - y_pred[i]) ** 2
                ss_tot += (y_test[i] - np.mean(y_test)) ** 2
            r2_error = 1 - ss_res / ss_tot
            square_error = np.sqrt(ss_res) / (len(y_test) - 2)
            return square_error, r2_error

    def predict(self, x_entry):
        if self.B is None:
            print("Find coefficients for model first")
            return None
        else:
            return np.dot(self.Coeff, x_entry.T) + self.B



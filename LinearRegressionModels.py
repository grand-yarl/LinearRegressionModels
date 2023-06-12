import numpy as np
import pandas as pd

"""
Ordinary Linear Regression class
attributes:
    Coeff - numpy array for regression coefficients list
    Free_coeff - free regression coefficient
    Square_error - statistic square error for model on train data
    R2_error - R2 error coefficient for model on train data
setting variables:
    number_of_singulars - number of singular values used to find pseudo inverse matrix
methods:
    static_methods:
        prepare_data - preparing full data from dataframe to test and train samples
    object_methods:
        fit_model - find model coefficients on train samples
        predict - find prediction values for input parameters
        test_error - calculate prediction error for test samples
"""
class LinearRegression:

    """
    Linear Regression class constructor
    inputs:
        number_of_singulars - number of singular values used to find pseudo inverse matrix
    """
    def __init__(self, number_of_singulars_=-1):
        self.Coeff = np.array([])
        self.Free_coeff = None
        self.Square_error = None
        self.R2_error = None
        self.number_of_singulars = number_of_singulars_

    """
    Data preparation method
    inputs:
        dataframe - pandas dataframe with full data
        x_names_list - names of model input parameters
        y_name - name of model output parameter
        test_size - size of test data sample (0.2 by default)
    outputs:
        x_train - train input sample
        y_train - train output sample
        x_test - test input sample
        y_test - test output sample
    """
    @staticmethod
    def prepare_data(dataframe, x_names_list, y_name, test_size=0.2):
        if (test_size > 1) or (test_size < 0):
            print("Test size must be from 0. to 1.")
            return None
        x = dataframe[x_names_list].to_numpy()
        y = dataframe[y_name].to_numpy()
        x_train, x_test, y_train, y_test = LinearRegression.__split_train_test__(x, y, test_size=test_size)
        return x_train, y_train, x_test, y_test

    """
    Method for splitting full data on train and test samples (private)
    inputs:
        x - full input data
        y - full output data
        test_size - size of test data sample (0.2 by default)
    outputs:
        x_train - train input sample
        y_train - train output sample
        x_test - test input sample
        y_test - test output sample    
    """
    @staticmethod
    def __split_train_test__(x, y, test_size=0.33):
        number_of_test = round(len(y) * test_size)
        number_of_train = len(y) - number_of_test
        x_train = x[0: number_of_train, :]
        x_test = x[number_of_train: len(y), :]
        y_train = y[0: number_of_train]
        y_test = y[number_of_train: len(y)]
        return x_train, x_test, y_train, y_test

    """
    Method for calculating model coefficients (Coeff and Free_coeff)
    inputs:
        x - train input sample
        y - train output sample
    """
    def fit_model(self, x, y):
        ones = np.array([np.ones(len(x))])
        x_new = np.append(x, ones.T, axis=1)
        x_pseudo = self.__pseudo_inv__(x_new)
        self.Coeff = np.dot(x_pseudo, y.T)
        self.Free_coeff = self.Coeff[len(self.Coeff) - 1]
        self.Coeff = np.delete(self.Coeff, len(self.Coeff) - 1)
        self.Square_error, self.R2_error = self.test_error(x, y)
        return

    """
    Method for calculating pseudo inverse matrix (private)
    input:
        matrix - matrix, from which pseudo inverse is calculating
    output:
        matrix_pseudo_inv - pseudo inverse matrix
    """
    def __pseudo_inv__(self, matrix):
        if self.number_of_singulars == -1:
            self.number_of_singulars = np.linalg.matrix_rank(matrix)
        u, s, vt = np.linalg.svd(matrix)
        s_inv = np.zeros(len(s))
        for i in range(self.number_of_singulars):
            if s[i] != 0:
                s_inv[i] = 1 / s[i]
        sigma = np.zeros(np.shape(matrix))
        sigma[:len(s), :len(s)] = np.diag(s_inv)
        matrix_pseudo_inv = np.dot(np.dot(vt.T, sigma.T), u.T)
        return matrix_pseudo_inv

    """
    Method for predicting output values
    input:
        x_entry - input parameters that are used for prediction
    output:
        y_predicted - list of predicted values
    """
    def predict(self, x_entry):
        if self.Free_coeff is None:
            print("Find coefficients for model first")
            return None
        else:
            y_predicted = np.dot(self.Coeff, x_entry.T) + self.Free_coeff
            return y_predicted

    """
    Method for calculating errors on test sample
    input:
        x_test - test input sample
        y_test - test output sample  
    output:
        square_error - square error on test sample
        r2_error - r2 error coefficient for test sample
    """
    def test_error(self, x_test, y_test):
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





# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
import matplotlib.pyplot as plt
from Project3.package.fitting_ANNs import fit_RNN, fit_LSTM
import seaborn as sns
from sklearn.metrics import mean_squared_error


def RNN_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=60,
                      n_hidden_layers=1, batch_size=1, activation='tanh',
                      random_state=None, verbose=0):
    """
    Function to perform a Time Series Cross-Validation with parameters'
    combination.

    Parameters:
    ===================
    :param data: pd.DataFrame: Containing daily samples and features.
    :param units: list: Containing values for units.
    :param epochs: list: Containing values for epochs.
    :param pred_feature: str: The name of the feature to be predicted.
    :param rolling: int: Containing the number of days before start
                         predicting.
    :param n_hidden_layers: int: Containing the number of hidden layers.
    :param batch_size: int: Containing the number of mini-batches.
    :param activation: str: The name of the activation function.
    :param random_state: int: The seeds.
    :param verbose: int: 0 for False and 1 for True.

    Returns:
    ===================
    avg_mse_train: numpy.array: Containing all training MSE.
    avg_mse_test: numpy.array: Containing all testing MSE.
    avg_mae_train: numpy.array: Containing all training MAE.
    avg_mae_test: numpy.array: Containing all testing MAE.
    y_pred_hist: numpy.array: Containing all predicted values in series.
    """
    # getting column that will be our target to predict
    pred_col_loc = data.columns.get_loc(pred_feature)

    # seeding
    np.random.seed(random_state)

    x = data.to_numpy()
    ini_point = -x.shape[0] + rolling

    # saving metrics
    avg_mse_train = np.zeros(shape=(len(epochs), len(units)))
    avg_mse_test = np.zeros(shape=(len(epochs), len(units)))
    avg_mae_train = np.zeros(shape=(len(epochs), len(units)))
    avg_mae_test = np.zeros(shape=(len(epochs), len(units)))
    y_pred_hist = np.empty(shape=(len(epochs), len(units)), dtype=object)

    # searching best parameter combinations
    for c_idx, u in enumerate(units):
        for r_idx, e in enumerate(epochs):

            mse_train, mse_test = [], []
            mae_train, mae_test = [], []
            _y_pred = []
            for i in range(ini_point, -1):  # -init_point to -2
                X_train = data.iloc[:i, :].to_numpy()
                Y_train = \
                    data.iloc[1:i + 1, pred_col_loc].to_numpy()[:, np.newaxis]

                model, history = fit_RNN(
                    X=X_train, Y=Y_train,
                    n_hidden_layers=n_hidden_layers, units=u,
                    epochs=e, batch_size=batch_size, activation=activation,
                    loss='mean_squared_error', optimizer='rmsprop',
                    random_state=random_state, verbose=verbose)

                mse_train.append(history.history['loss'][-1])
                mae_train.append(history.history['mean_absolute_error'][-1])

                X_validation = data.iloc[i, :].to_numpy()[np.newaxis, :]

                X_validation = \
                    np.reshape(
                        X_validation,
                        (X_validation.shape[0], 1, X_validation.shape[1]))

                y_pred = model.predict(X_validation)
                _y_pred.append(y_pred)

                Y_validation_true = \
                    np.array([data.iloc[i + 1, pred_col_loc]])[:, np.newaxis]

                mse_test.append(
                    mean_squared_error(y_true=Y_validation_true,
                                       y_pred=y_pred))

                mae_test.append(abs(float(Y_validation_true - y_pred)))

            avg_mse_train[r_idx, c_idx] = np.mean(mse_train)
            avg_mse_test[r_idx, c_idx] = np.mean(mse_test)
            avg_mae_train[r_idx, c_idx] = np.mean(mae_train)
            avg_mae_test[r_idx, c_idx] = np.mean(mae_test)
            y_pred_hist[r_idx, c_idx] = _y_pred

    return avg_mse_train, avg_mse_test, avg_mae_train, \
           avg_mae_test, y_pred_hist


def LSTM_CV_UNITxEPOCH(data, units, epochs, pred_feature='high', rolling=20,
                       n_hidden_layers=1, batch_size=1,
                       recurrent_activation='hard_sigmoid',
                       activation='tanh',
                       random_state=None, verbose=0):
    """
    Function to perform a Time Series Cross-Validation with parameters'
    combination.

    Parameters:
    ===================
    :param data: pd.DataFrame: Containing daily samples and features.
    :param units: list: Containing values for units.
    :param epochs: list: Containing values for epochs.
    :param pred_feature: str: The name of the feature to be predicted.
    :param rolling: int: Containing the number of days before start
                         predicting.
    :param n_hidden_layers: int: Containing the number of hidden layers.
    :param batch_size: int: Containing the number of mini-batches.
    :param recurrent_activation: str: The name of recurrent activation funct.
    :param activation: str: The name of the activation function.
    :param random_state: int: The seeds.
    :param verbose: int: 0 for False and 1 for True.

    Returns:
    ===================
    avg_mse_train: numpy.array: Containing all training MSE.
    avg_mse_test: numpy.array: Containing all testing MSE.
    avg_mae_train: numpy.array: Containing all training MAE.
    avg_mae_test: numpy.array: Containing all testing MAE.
    y_pred_hist: numpy.array: Containing all predicted values in series.
    """
    # getting column that will be our target to predict
    pred_col_loc = data.columns.get_loc(pred_feature)

    # seeding
    np.random.seed(random_state)

    x = data.to_numpy()
    ini_point = -x.shape[0] + rolling

    # saving metrics
    avg_mse_train = np.zeros(shape=(len(epochs), len(units)))
    avg_mse_test = np.zeros(shape=(len(epochs), len(units)))
    avg_mae_train = np.zeros(shape=(len(epochs), len(units)))
    avg_mae_test = np.zeros(shape=(len(epochs), len(units)))
    y_pred_hist = np.empty(shape=(len(epochs), len(units)), dtype=object)

    # searching best parameter combinations
    for c_idx, u in enumerate(units):
        for r_idx, e in enumerate(epochs):

            mse_train, mse_test = [], []
            mae_train, mae_test = [], []
            _y_pred = []
            for i in range(ini_point, -1):  # -init_point to -2
                X_train = data.iloc[:i, :].to_numpy()
                Y_train = \
                    data.iloc[1:i + 1, pred_col_loc].to_numpy()[:, np.newaxis]

                model, history = \
                    fit_LSTM(X=X_train, Y=Y_train,
                             n_hidden_layers=n_hidden_layers, units=u,
                             epochs=e, batch_size=batch_size,
                             activation=activation,
                             recurrent_activation=recurrent_activation,
                             loss='mean_squared_error',
                             optimizer='rmsprop', random_state=random_state,
                             verbose=verbose)

                mse_train.append(history.history['loss'][-1])
                mae_train.append(history.history['mean_absolute_error'][-1])

                X_validation = data.iloc[i, :].to_numpy()[np.newaxis, :]

                X_validation = \
                    np.reshape(
                        X_validation,
                        (X_validation.shape[0], 1, X_validation.shape[1]))

                y_pred = model.predict(X_validation)
                _y_pred.append(y_pred)

                Y_validation_true = \
                    np.array([data.iloc[i + 1, pred_col_loc]])[:, np.newaxis]

                mse_test.append(
                    mean_squared_error(y_true=Y_validation_true,
                                       y_pred=y_pred))

                mae_test.append(abs(float(Y_validation_true - y_pred)))

            avg_mse_train[r_idx, c_idx] = np.mean(mse_train)
            avg_mse_test[r_idx, c_idx] = np.mean(mse_test)
            avg_mae_train[r_idx, c_idx] = np.mean(mae_train)
            avg_mae_test[r_idx, c_idx] = np.mean(mae_test)
            y_pred_hist[r_idx, c_idx] = _y_pred

    return avg_mse_train, avg_mse_test, avg_mae_train, \
           avg_mae_test, y_pred_hist


def best_UNITxEPOCH(X, Y, units, epochs, n_hidden_layers=1,
                    activation='tanh', random_state=None, verbose=0):
    """
    Function to perform a parameters' combination.

    Parameters:
    ===================
    :param X: numpy.array: Containing samples and features.
    :param Y: numpy.array: Containing targets.
    :param units: list: Containing values for units.
    :param epochs: list: Containing values for epochs.
    :param n_hidden_layers: int: Containing the number of hidden layers.
    :param activation: str: The name of the activation function.
    :param random_state: int: The seeds.
    :param verbose: int: 0 for False and 1 for True.

    Returns:
    ===================
    mse_train: numpy.array: Containing all training MSE.
    mae_train: numpy.array: Containing all training MAE.
    """
    # seeding
    np.random.seed(random_state)

    # saving metrics
    mse_train = np.zeros(shape=(len(epochs), len(units)))
    mae_train = np.zeros(shape=(len(epochs), len(units)))

    # parameter combinations
    for c_idx, u in enumerate(units):
        for r_idx, e in enumerate(epochs):
            model_high, history = \
                fit_RNN(X, Y, n_hidden_layers=n_hidden_layers, units=u,
                        epochs=e, batch_size=X.shape[0],
                        activation=activation, loss='mean_squared_error',
                        optimizer='rmsprop', random_state=random_state,
                        verbose=verbose)
            mse_train[r_idx, c_idx] = \
                history.history['loss'][-1]
            mae_train[r_idx, c_idx] = \
                history.history['mean_absolute_error'][-1]

    # getting best arg for mse_train
    best_arg_mse_train = \
        np.unravel_index(np.argmin(mse_train, axis=None), mse_train.shape)
    best_row_mse_train = units[best_arg_mse_train[1]]
    best_col_mse_train = epochs[best_arg_mse_train[0]]

    print(f"Best loss is {mse_train[best_arg_mse_train]} with "
          f"{best_row_mse_train} unit and {best_col_mse_train} epochs.")

    # getting best arg for mse_train
    best_arg_mae_train = \
        np.unravel_index(np.argmin(mae_train, axis=None), mae_train.shape)
    best_row_mae_train = units[best_arg_mae_train[1]]
    best_col_mae_train = epochs[best_arg_mae_train[0]]

    print(f"Best MAE is {mae_train[best_arg_mae_train]} with "
          f"{best_row_mae_train} unit and {best_col_mae_train} epochs.")

    sns.heatmap(data=mse_train, xticklabels=units,
                yticklabels=epochs, annot=True,
                annot_kws={"size": 8.5}, fmt=".2f")
    plt.xlabel('Units')
    plt.ylabel('Epochs')
    plt.title(f'Last training MSE for activation {activation} '
              f'for {n_hidden_layers} hidden-layers.')
    plt.show()

    sns.heatmap(data=mae_train, xticklabels=units,
                yticklabels=epochs, annot=True,
                annot_kws={"size": 8.5}, fmt=".2f")
    plt.xlabel('Units')
    plt.ylabel('Epochs')
    plt.title(f'Last training MAE for activation {activation} '
              f'for {n_hidden_layers} hidden-layers.')
    plt.show()

    return mse_train, mae_train

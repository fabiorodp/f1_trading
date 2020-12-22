# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SimpleRNN
from keras import metrics
import numpy as np


def fit_RNN(X, Y, n_hidden_layers=1, units=100, epochs=50, batch_size=1,
            activation='tanh', loss='mean_squared_error', optimizer='rmsprop',
            random_state=None, verbose=0):
    """
    Function to create, compile and fit a SimpleRNN from Keras.

    Parameters:
    ===================
    :param X: numpy.array: Containing samples and features.
    :param Y: numpy.array: Containing targets.
    :param n_hidden_layers: int: The number of hidden layers.
    :param units: int: The number of units.
    :param epochs: int: The number of epochs.
    :param batch_size: int: The number of batches.
    :param activation: str: The name of the activation function.
    :param loss: str: The name of the loss-function.
    :param optimizer: str: The name of the optimaizer.
    :param random_state: int: The seeds.
    :param verbose: int: 0 for False and 1 for True.


    Returns:
    ===================
    model: keras object: Containing the trained model.
    history: keras object: Containing the metrics of the fitting.
    """
    # seeding
    np.random.seed(random_state)

    # reshaping to fit RNN
    # (n_samples, n_timestep, n_features=n_features)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # initializing sequential model
    model = Sequential()

    # adding input and hidden-layers on LSTM
    # input_shape = (n_time_step, n_features)
    if n_hidden_layers == 1:
        model.add(
            SimpleRNN(units, activation=activation,
                      input_shape=(1, X.shape[2])))

    elif n_hidden_layers == 2:
        model.add(
            SimpleRNN(units, activation=activation,
                      input_shape=(1, X.shape[2]),
                      return_sequences=True))
        model.add(SimpleRNN(units))

    else:
        model.add(
            SimpleRNN(units, activation=activation,
                      input_shape=(1, X.shape[2]),
                      return_sequences=True))

        for _ in range(n_hidden_layers - 2):
            model.add(SimpleRNN(units, return_sequences=True))

        model.add(SimpleRNN(units))

    # adding output layers
    model.add(Dense(1))

    # compiling
    model.compile(
        optimizer=optimizer, loss=loss,
        metrics=[metrics.mae, metrics.mean_absolute_percentage_error],
        loss_weights=None, weighted_metrics=None, run_eagerly=None)

    # fitting and training
    history = model.fit(
        x=X, y=Y, batch_size=batch_size, epochs=epochs,
        verbose=verbose, callbacks=None, validation_split=0.0,
        validation_data=None, shuffle=False, class_weight=None,
        sample_weight=None, initial_epoch=0, steps_per_epoch=None,
        validation_steps=None, validation_batch_size=None,
        validation_freq=1, max_queue_size=10, workers=1,
        use_multiprocessing=False)

    return model, history


def fit_LSTM(X, Y, n_hidden_layers=1, units=100, epochs=50, batch_size=1,
             activation='tanh', recurrent_activation='hard_sigmoid',
             loss='mean_squared_error', optimizer='rmsprop',
             random_state=None, verbose=0):
    """
    Function to create, compile and fit a SimpleRNN from Keras.

    Parameters:
    ===================
    :param X: numpy.array: Containing samples and features.
    :param Y: numpy.array: Containing targets.
    :param n_hidden_layers: int: The number of hidden layers.
    :param units: int: The number of units.
    :param epochs: int: The number of epochs.
    :param batch_size: int: The number of batches.
    :param activation: str: The name of the activation function.
    :param loss: str: The name of the loss-function.
    :param optimizer: str: The name of the optimaizer.
    :param random_state: int: The seeds.
    :param verbose: int: 0 for False and 1 for True.


    Returns:
    ===================
    model: keras object: Containing the trained model.
    history: keras object: Containing the metrics of the fitting.
    """
    # seeding
    np.random.seed(random_state)

    # reshaping to fit RNN
    # (n_samples, n_timestep, n_features=n_features)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # initializing sequential model
    model = Sequential()

    # adding input and hidden-layers on LSTM
    # input_shape = (n_time_step, n_features)
    if n_hidden_layers == 1:
        model.add(
            LSTM(units, activation=activation,
                 input_shape=(1, X.shape[2]),
                 recurrent_activation=recurrent_activation))

    elif n_hidden_layers == 2:
        model.add(
            LSTM(units, activation=activation,
                 input_shape=(1, X.shape[2]),
                 recurrent_activation=recurrent_activation,
                 return_sequences=True))
        model.add(LSTM(units))

    else:
        model.add(
            LSTM(units, activation=activation,
                 input_shape=(1, X.shape[2]),
                 recurrent_activation=recurrent_activation,
                 return_sequences=True))

        for _ in range(n_hidden_layers - 2):
            model.add(LSTM(units, return_sequences=True))

        model.add(LSTM(units))

    # adding output layers
    model.add(Dense(1))

    # compiling
    model.compile(
        optimizer=optimizer, loss=loss,
        metrics=[metrics.mae, metrics.mean_absolute_percentage_error],
        loss_weights=None, weighted_metrics=None, run_eagerly=None)

    # fitting and training
    history = model.fit(
        x=X, y=Y, batch_size=batch_size, epochs=epochs,
        verbose=verbose, callbacks=None, validation_split=0.0,
        validation_data=None, shuffle=False, class_weight=None,
        sample_weight=None, initial_epoch=0, steps_per_epoch=None,
        validation_steps=None, validation_batch_size=None,
        validation_freq=1, max_queue_size=10, workers=1,
        use_multiprocessing=False)

    return model, history

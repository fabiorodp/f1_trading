# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Dropout, TimeDistributed
from keras import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RollingOrigin:
    def __init__(self,
                 data,
                 pred_target='high',
                 random_state=None):

        # ########## seeding
        np.random.seed(random_state)
        # ##########

        # original data
        self.data = data

        # getting column that will be our target to predict
        self.pred_target_col_idx = self.data.columns.get_loc(pred_target)

        # to be returned
        self.y_pred = None

        # predicted outputs
        self.predicted_outputs = None

    def run(self,
            rolling=60,
            units=800,
            activation='tanh',
            loss='mean_squared_error',
            dropout=0.2,
            batch_size=1,
            epochs=10,
            verbose=1,
            shuffle=False):
        """
        Running the Time Series Cross-Validation.

        Parameters:
        ===================
        :param rolling: int:
        :param units: int: The number of neurons.
        :param activation: str: The name of the activation function.
        :param loss: str: The name of the loss function.
        :param dropout: float:
        :param batch_size: int: The number of batches.
        :param epochs: int: The number of epochs/interactions.
        :param verbose: int: If 0 it is False, else True.
        :param shuffle: bool:

        Return:
        ===================

        """
        # ########## not have leakage between train and test data
        # starts on row idx 0 to -2
        X = self.data.iloc[:-2, :].values

        # starts on row idx 1 to -1
        Y = self.data.iloc[1:-1, self.pred_target_col_idx].values

        # starts on row idx 1 to -1
        XV = self.data.iloc[1:-1, :].values

        # internal variables
        x_rolling, y_rolling, xv_rolling = None, None, None

        for idx, (x, y, xv) in enumerate(zip(X, Y, XV)):
            # fixing shapes
            x = x.reshape(1, -1)                # np.array of shape=(1, 17)
            y = np.array([y])[:, np.newaxis]    # np.array of shape=(1, 1)
            xv = xv[np.newaxis, :]              # np.array of shape=(1, 17)

            # storing rolling data for every time-step
            x_rolling = x if idx == 0 else np.concatenate((x_rolling, x),
                                                          axis=0)
            y_rolling = y if idx == 0 else np.concatenate((y_rolling, y),
                                                          axis=0)

            # reshaping to GPU
            X_train = np.reshape(x_rolling,
                                 (x_rolling.shape[0], 1, x_rolling.shape[1]))

            X_test = np.reshape(xv, (xv.shape[0], 1, xv.shape[1]))

            # training RNN
            model = Sequential()
            model.add(SimpleRNN(units=units, activation=activation,
                                input_shape=(1, X_train.shape[2]),
                                return_sequences=False))
            # model.add(Dropout(dropout))
            # model.add(TimeDistributed(Dense(1)))
            model.add(Dense(1))
            model.compile(optimizer='rmsprop', loss=loss,
                          metrics=[metrics.mae], loss_weights=None,
                          weighted_metrics=None, run_eagerly=None)

            model.fit(x=X_train, y=y_rolling, batch_size=batch_size,
                      epochs=epochs, verbose=verbose, callbacks=None,
                      validation_split=0.0, validation_data=None,
                      shuffle=shuffle, class_weight=None,
                      sample_weight=None, initial_epoch=0,
                      steps_per_epoch=None, validation_steps=None,
                      validation_batch_size=None, validation_freq=1,
                      max_queue_size=10, workers=1,
                      use_multiprocessing=True)

            # cross-validating
            y_hat = model.predict(X_test)

            # storing predicted output
            self.y_pred = y_hat if idx == 0 else \
                np.concatenate((self.y_pred, y_hat), axis=0)

        # printing the validated accuracy starting from the 5th prediction
        self.accuracies(starting_point_idx=5)

    def accuracies(self, starting_point_idx=5):
        """
        Returns metrics for given start point, i. e., best validation for
        the last 10 days (-10) or best overall validation starting from 5
        (default).

        Parameters:
        ===================
        :param starting_point_idx: int: The index number of the starting
                                        point for checking the accuracy
                                        validation.
        """
        y_true = self.data.iloc[2:, self.pred_target_col_idx].values

        print('Test MSE: ',
              mean_squared_error(y_pred=self.y_pred[starting_point_idx:],
                                 y_true=y_true[starting_point_idx:]))
        print('Test MAE: ',
              mean_absolute_error(y_pred=self.y_pred[starting_point_idx:],
                                  y_true=y_true[starting_point_idx:]))

    def add_new_features(self, new_feature_name='Predicted output'):
        """
        Saving new features as dataframe.

        Parameters:
        ===================
        :param new_feature_name: pandas.DataFrame: The name of the new
                                                   feature.
        """
        y_pred_dates = self.data.index[2:]

        if self.predicted_outputs is None:
            self.predicted_outputs = \
                pd.DataFrame(y_pred_dates, columns=['Date'])
            self.predicted_outputs[f'{new_feature_name}'] = self.y_pred

        else:
            self.predicted_outputs[f'{new_feature_name}'] = self.y_pred

    def save_csv(self, name='new_features'):
        """
        Saving the new features in CSV file.

        Parameters:
        ===================
        :param name: str: The path/name of the csv file.
        """
        self.predicted_outputs.to_csv(
            f'{name}.csv', sep=';', index_label=False)


if __name__ == '__main__':
    pass

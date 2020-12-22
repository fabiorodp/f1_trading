# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
from Project3.package.studies import RNN_CV_UNITxEPOCH


def get_predicted_features(data_1D, units=[800], epochs=[10],
                           rolling=60, out_csv=None):
    """
    Function to generate predictions for feature creation.

    Parameters:
    ===================
    :param data_1D: pandas.DataFrame: Containing samples and features.
    :param units: list: Containing the number of units.
    :param epochs: list: Containing the number of epochs.
    :param rolling: int: Number of days to train the model before predict.
    :param out_csv: str: The path and file name to save the predicted
                         features.

    Returns:
    ===================
    y_hat_highest: list: Containing all predictions in series for highest.
    y_hat_lowest: list: Containing all predictions in series for lowest.
    """
    # ########### training RNN model for high
    mse_train_h_t1, mse_test_h_t1, mae_train_h_t1, mae_test_h_t1, y_pred_h = \
        RNN_CV_UNITxEPOCH(data_1D, units, epochs,
                          pred_feature='high', rolling=rolling,
                          n_hidden_layers=1, batch_size=1, activation='tanh',
                          random_state=10, verbose=1)

    # ########### training RNN model for low
    mse_train_l_t1, mse_test_l_t1, mae_train_l_t1, mae_test_l_t1, y_pred_l = \
        RNN_CV_UNITxEPOCH(data_1D, units, epochs,
                          pred_feature='low', rolling=60,
                          n_hidden_layers=1, batch_size=1, activation='tanh',
                          random_state=10, verbose=1)

    # ########### getting predicted features from RNN
    y_hat_highest = [float(i) for i in y_pred_h[0][0]]
    y_hat_lowest = [float(i) for i in y_pred_l[0][0]]

    if out_csv is not None:
        df = pd.DataFrame(y_hat_highest, columns=['Predicted High'])
        df['Predicted Low'] = y_hat_lowest
        df.to_csv(f'{out_csv}', sep=';', index_label=False)

    return y_hat_highest, y_hat_lowest

# -*- coding: utf-8 -*-

"""
Helper functions to handle with feature engineering.
"""

__author__ = "Fábio Rodrigues Pereira"
__email__ = "fabio@fabiorodriguespereira.com"

import pandas as pd
import numpy as np
from datetime import datetime


def day_data_engineering(df):
    # y1 = df.iloc[1:, 5].values - df.iloc[1:, 2].values  # Return
    # y2 = np.where(y1 >= 0, 1, 0)  # ReturnBin
    # y3 = np.where(y1 <= -50, 1, 0)  # RN5Bin
    # y4 = np.where(y1 >= 50, 1, 0)  # RP5Bin
    x = df.iloc[1:, 0]  # Date
    x1 = df.iloc[1:, 1]  # Open
    # x2 = df.iloc[1:, 1] - df.iloc[:-1, 4]  # Gap: (Open - LastClose)
    x3 = df.iloc[:-1, 1].values  # LastOpen
    x4 = df.iloc[:-1, 2].values  # LastHigh
    x5 = df.iloc[:-1, 3].values  # LastLow
    x6 = df.iloc[:-1, 4].values  # LastClose
    x7 = df.iloc[:-1, 5].values  # LastVolume
    x8 = df.iloc[:-1, 4] - df.iloc[:-1, 1]  # LastReturn: (LastClose -
    # LastOpen)

    # x3 = df.iloc[:-1, 3].values - df.iloc[:-1, 4].values  # LastAmplitude
    # x5 = df.iloc[:-1, 3].values - df.iloc[:-1, 2].values  # LastDrawUP
    # x7 = df.iloc[:-1, 4].values - df.iloc[:-1, 2].values  # LastDrawDown
    # x9 = df.iloc[:-1, 5].values - df.iloc[:-1, 2].values  # LastReturn

    df1 = np.vstack(
        [x, x1, x3, x4, x5, x6, x7, x8]).T

    df1 = pd.DataFrame(df1, columns=[
        'Date', 'Open', 'LastOpen', 'LastHigh',
        'LastLow', 'LastClose', 'LastVolume', 'LastReturn'])

    return df1


def tick_data_engineering(df):
    y1 = df.iloc[1:, 5].values - df.iloc[1:, 2].values  # Return
    y2 = np.where(y1 >= 0, 1, 0)  # ReturnBin
    y3 = np.where(y1 <= -50, 1, 0)  # RN5Bin
    y4 = np.where(y1 >= 50, 1, 0)  # RP5Bin
    x1 = df.iloc[1:, 2].values - df.iloc[:-1, 5].values  # OvernightReturn
    x2 = df.iloc[1:, 2].values
    x3 = df.iloc[:-1, 3].values - df.iloc[:-1, 4].values  # LastAmplitude
    x4 = df.iloc[:-1, 2].values  # LastOpen
    x5 = df.iloc[:-1, 3].values - df.iloc[:-1, 2].values  # LastDrawUP
    x6 = df.iloc[:-1, 3].values  # LastHigh
    x7 = df.iloc[:-1, 4].values - df.iloc[:-1, 2].values  # LastDrawDown
    x8 = df.iloc[:-1, 4].values  # LastLow
    x9 = df.iloc[:-1, 5].values - df.iloc[:-1, 2].values  # LastReturn
    x10 = df.iloc[:-1, 5].values  # LastClose
    x11 = df.iloc[:-1, 6].values  # LastVolume

    df1 = np.vstack(
        [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y1, y2, y3,
         y4]).T

    # Deleting day with zero volume (error):
    df1 = np.delete(df1, np.argwhere(x11 == 0), axis=0)

    df1 = pd.DataFrame(df1, columns=[
        'Gap', 'Open', 'LastAmplitude', 'LastOpen', 'LastDrawUP',
        'LastHigh', 'LastDrawDown', 'LastLow', 'LastReturn', 'LastClose',
        'LastVolume', 'Return', 'ReturnBin', 'RN5Bin', 'RP5Bin'])

    return df1

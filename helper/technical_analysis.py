# -*- coding: utf-8 -*-

"""
Helper functions to create technical analysis instruments.
"""

__author__ = "Fábio Rodrigues Pereira"
__email__ = "fabio@fabiorodriguespereira.com"

import pandas as pd
import numpy as np


def RSI(data, time_range):
    """Creates the RSI indicator."""
    diff = data.diff(1).dropna()  # diff in one field(one day)

    # This preservers dimensions off diff values:
    up_chg = 0 * diff
    down_chg = 0 * diff

    # Up change is equal to the positive difference, otherwise equal to zero:
    up_chg[diff > 0] = diff[diff > 0]

    # Down change is equal to negative difference, otherwise equal to zero:
    down_chg[diff < 0] = diff[diff < 0]

    up_chg_avg = up_chg.ewm(com=time_range - 1,
                            min_periods=time_range).mean()

    down_chg_avg = down_chg.ewm(com=time_range - 1,
                                min_periods=time_range).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def SMA(data, time_range):
    """Creates the SMA indicator."""
    sma = data.rolling(time_range).mean()
    return sma

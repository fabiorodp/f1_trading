# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import pandas as pd
import numpy as np
from Project3.package.technical_analysis import bollinger_bands, ema
from Project3.package.time_series_cv import RollingOrigin
from Project3.package.trade_systems import algo_trading
from Project3.package.engineering_features import get_predicted_features


# ########### reading data daily
data = pd.read_csv(
    filepath_or_buffer='Project3/data_all_features.csv',
    sep=';'
)

# ########### reading 15min OLHCV data
data_15min = pd.read_csv(
    filepath_or_buffer='Project3/data/PETR4_15min_OLHCV.csv',
    sep=';'
)

# slipage
slipage = 0.05

# defining the first and last trading days
frist_day, last_day = data.index[0], data.index[-1]
data_15min = data_15min.loc[frist_day:last_day]

# storing all and triggered positions
all_triggered_trades = pd.DataFrame(columns=['time', 'position'])

for day in data.iterrows():                # loop throughout day
    current_day = day[0]
    pred_low_idx = data.columns.get_loc('Predicted Low')
    pred_low = day[1][pred_low_idx]
    pred_high_idx = data.columns.get_loc('Predicted High')
    pred_high = day[1][pred_high_idx]
    close_price_idx = data.columns.get_loc('close')
    close_price = day[1][close_price_idx]
    open_price_idx = data.columns.get_loc('open')
    open_price = day[1][open_price_idx]

    # skip strategy if the price already opens outside of the predicted range
    if (open_price > pred_high) or (open_price < pred_low):
        continue

    # storing all triggers
    triggered_trades = {'time': [], 'position': []}

    for time in data_15min.iterrows():     # loop throughout time
        current_day_time = time[0]
        low_idx = data_15min.columns.get_loc('low')
        low = time[1][low_idx]
        high_idx = data_15min.columns.get_loc('high')
        high = time[1][high_idx]

        if current_day_time[:10] == current_day:

            # it is the first position of the day
            if len(triggered_trades['position']) == 0:

                # opened a bearish position
                if high > pred_high + slipage:
                    triggered_trades['time'].append(current_day_time)
                    triggered_trades['position'].append(pred_high + slipage)

                # opened a bullish position
                if low < pred_low - slipage:
                    triggered_trades['time'].append(current_day_time)
                    triggered_trades['position'].append(-pred_low - slipage)

            # there exists prior position in the day
            else:

                # there is a opened position
                if len(triggered_trades['position']) % 2 != 0:

                    # the opened position is bearish
                    if triggered_trades['position'][-1] > 0:

                        if low < pred_low - slipage:
                            # close position if buy is triggered
                            triggered_trades['time'].append(current_day_time)
                            triggered_trades['position'].append(
                                -pred_low - slipage)

                            # open position if buy is triggered
                            triggered_trades['time'].append(current_day_time)
                            triggered_trades['position'].append(
                                -pred_low - slipage)

                    # the opened position is bullish
                    elif triggered_trades['position'][-1] < 0:

                        if high > pred_high + slipage:
                            # close position if sell is triggered
                            triggered_trades['time'].append(current_day_time)
                            triggered_trades['position'].append(
                                pred_high + slipage)

                            # open position if sell is triggered
                            triggered_trades['time'].append(current_day_time)
                            triggered_trades['position'].append(
                                pred_high + slipage)
        else:
            # there is a opened position by the end of the day
            if len(triggered_trades['position']) % 2 != 0:

                # the opened position is bearish
                if triggered_trades['position'][-1] > 0:

                    # close position by the end of the day
                    triggered_trades['time'].append(current_day + ' 18:00:00')
                    triggered_trades['position'].append(-close_price)

                # the opened position is bullish
                elif triggered_trades['position'][-1] < 0:

                    # close position by the end of the day
                    triggered_trades['time'].append(current_day + ' 18:00:00')
                    triggered_trades['position'].append(close_price)

    # concatenate all results
    all_triggered_trades = \
        pd.concat([all_triggered_trades, pd.DataFrame(triggered_trades)])

# adding cumulative sum
all_triggered_trades['cumsum'] = all_triggered_trades['position'].cumsum()

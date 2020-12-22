# UiO: FYS-STK4155 - H20
# Project 3
# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no


def algo_trading(data_1D, data_15min, strategy='5 BB'):
    """
    Function to reproduce the Bollinger Bands Algorithm trading strategy.

    Parameters:
    ===================
    :param data_1D: pd.DataFrame: Containing all OLHCV and other
                                  features of a daily's periodicity.
    :param data_15min: pd.DataFrame: Containing all OLHCV and other
                                     features of a 15min's periodicity.
    :param strategy: str: The name of the strategy. Default: '5 BB' for 5
                          SMA Bollinger Bands. Option: 'Our' for our algo
                          trading system.

    Returns:
    ===================
    all_valid_trades: dict: Containing all valid positions and DateTime.
    all_triggered_trades dict: Containing all triggered positions and
                               DateTime.
    """
    # defining the first and last trading days
    frist_day, last_day = data_1D.index[0], data_1D.index[-1]
    data_15min = data_15min.loc[frist_day:last_day]

    # that are what we are looking for in the end of the strategy
    all_valid_trades = {'time': [], 'position': []}
    all_triggered_trades = {'time': [], 'position': []}

    # for each day trading session
    for i in range(1, len(data_1D)):
        start, end = data_1D.index[i - 1], data_1D.index[i]
        # start, end = '2020-07-21', '2020-07-22'  # testing

        # joining all features to data_15min
        for col in data_1D.columns[5:]:
            data_15min.loc[start:end, col] = data_1D.loc[start, col]

        # getting individually day trading session
        day_session = data_15min.loc[start:end]

        # storing all triggers
        triggered_trades = {'time': [], 'position': []}

        # getting actual_candle and closing_price
        actual_candle, closing_price = None, None

        # getting all triggers in a day trading session
        for candle_range in day_session.iterrows():
            # getting DateTime of the actual_candle
            actual_candle = candle_range[0]

            # getting values for the highest and lowest of the actual_candle
            highest = candle_range[1][1]
            lowest = candle_range[1][2]

            # getting value for the upper and lower areas
            _Upper, _Lower = None, None
            if strategy == '5 BB':
                _Upper = candle_range[1][6]
                _Lower = candle_range[1][7]

            elif strategy == 'Our':
                _Upper = candle_range[1][17]
                _Lower = candle_range[1][18]

            # getting the closing value for the actual_candle
            closing_price = candle_range[1][3]

            # when triggered, we sell at the BB_Upper's price
            if highest > _Upper:
                triggered_trades['position'].append(_Upper)
                triggered_trades['time'].append(f'{actual_candle}')

            # when triggered, we buy at the BB_Lower's price
            if lowest < _Lower:
                triggered_trades['position'].append(-_Lower)
                triggered_trades['time'].append(f'{actual_candle}')

        # go to the next day trading session if no triggers occurred
        if len(triggered_trades['position']) == 0:
            continue

        # filter triggers for only 1 trade per time
        else:
            # storing triggered_trades
            all_triggered_trades['time'].append(
                triggered_trades['time'])

            all_triggered_trades['position'].append(
                triggered_trades['position'])

            # getting only valid trades
            valid_trades = \
                {'time': [triggered_trades['time'][0]],
                 'position': [triggered_trades['position'][0]]}

            # looping throughout all trades and storing only the valid ones
            for t in range(1, len(triggered_trades['position'])):

                # 2 selling trades in a row are not allowed
                if (valid_trades['position'][-1] > 0) \
                        and (triggered_trades['position'][t] > 0):
                    continue

                # 2 buying trades in a row are not allowed
                elif (valid_trades['position'][-1] < 0) \
                        and (triggered_trades['position'][t] < 0):
                    continue

                # only trades in opposite sides are allowed
                else:
                    # storing time and position values for valid trades
                    valid_trades['time'].append(
                        triggered_trades['time'][t])

                    valid_trades['position'].append(
                        triggered_trades['position'][t])

            # closing position in the end of the day trading session
            if len(valid_trades['position']) % 2 != 0:
                # storing time of the closing price
                valid_trades['time'].append(actual_candle)

                if valid_trades['position'][-1] > 0:
                    # storing closing_price when the previous position
                    # was selling
                    valid_trades['position'].append(-closing_price)

                else:
                    # storing closing_price when the previous position
                    # was buying
                    valid_trades['position'].append(closing_price)

            # storing all valid trades
            all_valid_trades['time'].append(valid_trades['time'])
            all_valid_trades['position'].append(valid_trades['position'])

    return all_valid_trades, all_triggered_trades

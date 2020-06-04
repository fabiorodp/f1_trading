# -*- coding: utf-8 -*-

"""
Helper functions to download historical quotation data of companies listed
at B3 (BMF/BOVESPA).
"""

__author__ = "Fábio Rodrigues Pereira"
__email__ = "fabio@fabiorodriguespereira.com"

import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract


# ===========================================================================
# Handling B3 tick data:
# ===========================================================================
def read_bmf_txt(path_file, separator, ticker, ticker_and_symbols=False):
    """
    -------------------------------------------------------------
    Column                Initial Position   Length   Description
    -------------------------------------------------------------
    *Session Date                         1       10   Session date
    *Instrument Symbol                   12       50   Instrument identifier
    Trade Number                        63       10   Trade number
    *Trade Price                         74       20   Trade price
    *Traded Quantity                     95       18   Traded quantity
    *Trade Time                         114       15   Trade time (format HH:MM:SS.NNNNNN)
    **Trade Indicator                    127        1   Trade indicador: 1 - Trade  / 2 - Trade cancelled
    Buy Order Date                     129       10   Buy order date
    Sequential Buy Order Number        140       15   Sequential buy order number
    Secondary Order ID - Buy Order     156       15   Secondary Order ID -  Buy Order.
    *Aggressor Buy Order Indicator      172        1   0 - Neutral (Order was not executed) / 1 - Aggressor / 2 - Passive
    Sell Order Date                    174       10   Sell order sell date
    Sequential Sell Order Number       185       15   Sequential sell order number
    Secondary Order ID - Sell Order    201       15   Secondary Order ID -  Buy Order.
    *Aggressor Sell Order Indicator     217        1   0 - Neutral (Order was not executed) / 1 - Aggressor / 2 - Passive
    Cross Trade Indicator              219        1   Define if the cross trade was intentional: 1 - Intentional / 0 - Not Intentional
    Buy Member                         221        8   Entering Firm (Buy Side) - Available from March/2014
    Sell Member                        230        8   Entering Firm (Sell Side) - Available from March/2014

    Obs: Delimiter of details columns ';' (semilocon)

    print(symbols)
    ['AFSM20', 'AFSN20', 'AUDM20', 'AUDN20', 'AUSM20', 'AUSN20', 'AUSQ20',
    'B3SAOM20', 'B3SAON20', 'BGIK20', 'BGIK20C018500', 'BGIK20C019500',
    'BGIK20P020000', 'BGIK20P020200', 'BGIK20P021000', 'BGIK20P021200',
    'BGIK20P021500', 'BGIK21', 'BGIM20', 'BGIM20C020000', 'BGIM20C020400',
    'BGIM20P019500', 'BGIN20', 'BGIN20C019200', 'BGIN20C021000',
    'BGIN20P019000', 'BGIN20P020000', 'BGIN20P021200', 'BGIQ20', 'BGIV20',
    'BGIV20C019500', 'BGIV20C019650', 'BGIV20C020600', 'BGIV20C021000',
    'BGIV20C021500', 'BGIV20C022000', 'BGIV20P019500', 'BGIV20P019650',
    'BGIV20P020500', 'BGIV20P021000', 'BGIV20P021500', 'BGIX20', 'BRIM20',
    'BRIQ20', 'CADM20', 'CADN20', 'CANM20', 'CANQ20', 'CCMF21', 'CCMH21',
    'CCMN20', 'CCMN20P004500', 'CCMU20', 'CCMX20', 'CCROOM20', 'CCROON20',
    'CHFM20', 'CHFN20', 'CIELOM20', 'CIELON20', 'CLPQ20', 'CMIGPM20',
    'CMIGPN20', 'CNHM20', 'CNHN20', 'CNHU20', 'COGNOM20', 'COGNON20',
    'CPMM20C099000', 'CPMM20C099500', 'DAPF21', 'DAPK21', 'DAPK23',
    'DAPK25', 'DAPQ20', 'DAPQ22', 'DAPQ24', 'DAPQ26', 'DAPQ28', 'DAPQ30',
    'DDIM20', 'DDIN20', 'DI1F21', 'DI1F22', 'DI1F23', 'DI1F24', 'DI1F25',
    'DI1F26', 'DI1F27', 'DI1F28', 'DI1F29', 'DI1F31', 'DI1H21', 'DI1J21',
    'DI1J22', 'DI1J23', 'DI1J24', 'DI1J25', 'DI1K21', 'DI1M20', 'DI1N20',
    'DI1N21', 'DI1N22', 'DI1N23', 'DI1N24', 'DI1Q20', 'DI1U20', 'DI1V20',
    'DI1V21', 'DI1V22', 'DI1V23', 'DI1X20', 'DI1Z20', 'DOLG21P005200',
    'DOLM20', 'DOLM20C005325', 'DOLM20C005500', 'DOLM20C005700',
    'DOLM20C005800', 'DOLM20C005900', 'DOLM20C006000', 'DOLM20P005200',
    'DOLM20P005250', 'DOLM20P005300', 'DOLM20P005350', 'DOLM20P005400',
    'DOLM20P005500', 'DOLM20P005600', 'DOLN20', 'DOLN20C005200',
    'DOLN20C005400', 'DOLN20C005450', 'DOLN20C005475', 'DOLN20C005500',
    'DOLN20C005600', 'DOLN20C005700', 'DOLN20C005750', 'DOLN20C005900',
    'DOLN20C006000', 'DOLN20C006100', 'DOLN20C006200', 'DOLN20C006400',
    'DOLN20C006600', 'DOLN20P004250', 'DOLN20P004300', 'DOLN20P004350',
    'DOLN20P004400', 'DOLN20P004800', 'DOLN20P004900', 'DOLN20P005000',
    'DOLN20P005050', 'DOLN20P005150', 'DOLN20P005200', 'DOLN20P005300',
    'DOLN20P005350', 'DOLN20P005400', 'DOLN20P005450', 'DOLN21P004500',
    'DOLQ20', 'DOLQ20C005500', 'DOLQ20C005600', 'DOLQ20P005200',
    'DOLU20P005200', 'DOLV20', 'DOLZ20C004800', 'DOLZ20C006050',
    'DOLZ20P004700', 'DOLZ20P004800', 'DOLZ20P005000', 'DOLZ20P006050',
    'DR1M20N20', 'DR1M20Q20', 'ETHF21', 'ETHN20', 'ETHQ20', 'ETHU20',
    'ETHV20', 'ETHX20', 'ETHZ20', 'EUPM20', 'EUPN20', 'EUPQ20', 'EURM20',
    'EURN20', 'EURQ20', 'FRCF21', 'FRCF22', 'FRCF23', 'FRCF24', 'FRCF25',
    'FRCF26', 'FRCG21', 'FRCH21', 'FRCJ21', 'FRCJ22', 'FRCJ23', 'FRCK21',
    'FRCN20', 'FRCN21', 'FRCN23', 'FRCN24', 'FRCQ20', 'FRCU20', 'FRCV20',
    'FRCV21', 'FRCV22', 'FRCV24', 'FRCX20', 'FRCZ20', 'FRP0', 'GBPM20',
    'GBPN20', 'GBRM20', 'GBRN20', 'GBRU20', 'HYPEOM20', 'HYPEON20',
    'ICFH21', 'ICFU20', 'ICFU20C012250', 'ICFU20P011250', 'ICFZ20',
    'IDIF21C287900', 'IDIF21C288300', 'IDIF21C288400', 'IDIF21C291700',
    'IDIF21C295700', 'IDIF21P286100', 'IDIF21P286200', 'IDIF21P286300',
    'IDIF21P286400', 'IDIF21P286500', 'IDIF21P286600', 'IDIF21P286700',
    'IDIF21P287400', 'IDIF21P287500', 'IDIF21P287800', 'IDIF21P287900',
    'IDIF21P288000', 'IDIF21P288400', 'IDIF21P288500', 'IDIF21P289200',
    'IDIF21P289300', 'IDIF21P289400', 'IDIF21P289700', 'IDIF21P289800',
    'IDIF21P290400', 'IDIF21P290500', 'IDIF21P290600', 'IDIF21P290700',
    'IDIF21P291000', 'IDIF21P291100', 'IDIF21P291200', 'IDIF21P291300',
    'IDIF21P291700', 'IDIF21P291800', 'IDIF21P291900', 'IDIN20C285300',
    'IDIN20C285400', 'IDIN20C285500', 'IDIN20C285600', 'IDIN20C285900',
    'IDIN20P284400', 'IDIN20P284500', 'IDIN20P284600', 'IDIN20P284700',
    'IDIN20P284800', 'IDIN20P284900', 'IDIN20P285000', 'IDIN20P285100',
    'IDIN20P285200', 'IDIN20P285300', 'INDM20', 'ISPM20', 'ISPN20P002850',
    'ISPU20', 'ISPZ20C002900', 'ISPZ20C003100', 'JAPM20', 'JAPN20',
    'JAPQ20', 'JPYM20', 'JPYN20', 'MEXM20', 'MEXN20', 'MEXQ20', 'MXNM20',
    'MXNN20', 'NZDM20', 'NZDN20', 'OZ1D', 'OZ2D', 'OZ3D', 'PCAROM20',
    'PCARON20', 'PETRPM20', 'PETRPN20', 'PSSAOM20', 'PSSAON20', 'RUBN20',
    'SJCN20', 'SJCQ20', 'SWIM20', 'SWIN20', 'T10M20', 'T10U20', 'TRYM20',
    'TRYN20', 'USIMAM20', 'USIMAN20', 'VALEOM20', 'VALEON20', 'VVAROM20',
    'VVARON20', 'WD1M20N20', 'WDOF21', 'WDOM20', 'WDON20', 'WDON20C005350',
    'WDON20C005375', 'WDON20C005400', 'WDOQ20', 'WDOZ20', 'WINM20',
    'WINQ20', 'WSPM20', 'WSPU20', <NA>]
    """
    # Reading database
    df = pd.read_csv(path_file, sep=separator, header=None)

    # Defining the header of the dataset:
    columns = ['Session Date', 'Instrument Symbol', 'Trade Number',
               'Trade Price', 'Traded Quantity', 'Trade Time',
               'Trade Indicator', 'Buy Order Date',
               'Sequential Buy Order Number',
               'Secondary Order ID - Buy Order',
               'Aggressor Buy Order Indicator',
               'Sell Order Date', 'Sequential Sell Order Number',
               'Secondary Order ID - Sell Order',
               'Aggressor Sell Order Indicator', 'Cross Trade Indicator',
               'Buy Member', 'Sell Member']

    # Setting the header:
    df.columns = columns

    # Converting object columns in string type:
    df['Session Date'] = df['Session Date'].astype('string')
    df['Instrument Symbol'] = df['Instrument Symbol'].astype('string')
    df['Trade Time'] = df['Trade Time'].astype('string')

    # Dropping indexes:
    drop_index = df['Trade Indicator'][df['Trade Indicator'] == 2].index
    df.drop(drop_index, axis=0, inplace=True)

    drop_index = df['Aggressor Buy Order Indicator'][
        df['Aggressor Buy Order Indicator'] == 0].index
    df.drop(drop_index, axis=0, inplace=True)

    drop_index = df['Aggressor Sell Order Indicator'][
        df['Aggressor Sell Order Indicator'] == 0].index
    df.drop(drop_index, axis=0, inplace=True)

    # Columns to drop:
    drop_columns = ['Trade Number',
                    'Buy Order Date',
                    'Sequential Buy Order Number',
                    'Secondary Order ID - Buy Order',
                    'Sell Order Date', 'Sequential Sell Order Number',
                    'Secondary Order ID - Sell Order',
                    'Cross Trade Indicator',
                    'Buy Member', 'Sell Member']

    df.drop(drop_columns, axis=1, inplace=True)

    if ticker is None:
        symbols = list(pd.unique(df['Instrument Symbol']))
        return symbols

    if ticker_and_symbols is True:
        symbols = list(pd.unique(df['Instrument Symbol']))
        df_ticker = df.groupby(['Instrument Symbol']).get_group(ticker)
        return df_ticker, symbols

    else:
        df_ticker = df.groupby(['Instrument Symbol']).get_group(ticker)
        return df_ticker


def create_tick_candles(data_path, tick_range):
    df = pd.read_csv(data_path, index_col='Unnamed: 0')

    d = {'date': [], 'time': [], 'open': [], 'low': [], 'high': [],
         'close': [], 'volume': [], 'mean': [], 'std': []}

    for i in range(tick_range, df.shape[0], tick_range):
        """Error: when it updates the cycles it counts since the beginning"""
        d['date'].append(df.iloc[i, 0])
        d['time'].append(df.iloc[i, 4])
        d['open'].append(df.iloc[i - tick_range, 2])
        d['low'].append(df.iloc[i - tick_range:i, 2].min())
        d['high'].append(df.iloc[i - tick_range:i, 2].max())
        d['close'].append(df.iloc[i, 2])
        d['volume'].append(df.iloc[i - tick_range:i, 3].sum())
        d['mean'].append(df.iloc[i - tick_range:i, 2].mean())
        d['std'].append(df.iloc[i - tick_range:i, 2].std())

    return pd.DataFrame(d)


def _convert_time(time):
    """Convert Unix timestamp to normal date/hour format"""
    ts = int(time)
    # df1.index = pd.to_datetime(df1.index, unit='s')
    return datetime.utcfromtimestamp(ts).strftime('%Y%m%d %H:%M:%S')

'''
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def historicalData(self, reqId, bar):
        api.data.append([bar.date, bar.open, bar.high, bar.low,
                         bar.close, bar.volume])

    def historicalTicksLast(self, reqId, ticks, done):
        for tick in ticks:
            api.ticks.append([tick.time, tick.price, tick.size])


def run_loop():
    api.run()'''


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.ticks = []

    def historicalData(self, reqId, bar):
        self.data.append([bar.date, bar.open, bar.high, bar.low,
                         bar.close, bar.volume])

    def historicalTicksLast(self, reqId, ticks, done):
        for tick in ticks:
            self.ticks.append([tick.time, tick.price, tick.size])

    def run_loop(self):
        self.run()

    def download_hist_data(self,
            tickerID, ticker, sectype, exchange, currency, end_period,
            interval, timePeriod, datatype, RTH=0, timeFormat=1,
            streaming=False, ip='127.0.0.1', port=4001, id=0):

        self.connect(ip, port, id)

        # Start the socket in a thread:
        api_thread = threading.Thread(target=self.run(), daemon=True)
        api_thread.start()

        time.sleep(1)

        # Creating contract object:
        c = Contract()
        c.symbol = ticker
        c.secType = sectype
        c.exchange = exchange
        c.currency = currency

        self.reqHistoricalData(
            tickerID,  # Ticker ID
            c,  # Contract
            end_period,  # End Date
            interval,  # '12 M' - Interval
            timePeriod,  # '1 M' -  # Time Period
            datatype,  # 'TRADES' - Data Type  # ADJUSTED_LAST
            RTH,  # If pre-market data, set this to 1
            timeFormat,  # Time Format: 1 for readable time and 2 for Epcoh
            streaming,  # Streaming: if True updates every 5 seconds
            [])

        time.sleep(1)

        # Working with Pandas DataFrames:
        df = pd.DataFrame(
            self.data,
            columns=
            ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])

        self.disconnect()

        return df


"""
app = IBapi()
app.connect('127.0.0.1', 7496, 0)  # 4001

# Start the socket in a thread:
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

# Sleep interval to allow time for connection to server:
time.sleep(1)

# Create contract object:
contract = Contract()
contract.symbol = 'PETR4'
contract.secType = 'STK'
contract.exchange = 'BOVESPA'
contract.currency = 'BRL'

# Initialize variable to store candle:
app.data = []
app.tick = []

# Request historical candles:
app.reqHistoricalData(
    1,
    contract,
    '',  # End Date
    '12 M',  # Interval
    '1 M',  # Time Period
    'TRADES',  # Data Type  # ADJUSTED_LAST
    0,  # RTH. If pre-market data, set this to 1
    1,  # Time Format: 1 for readable time and 2 for Epcoh
    False,  # Streaming: if True updates every 5 seconds
    [])

app.reqHistoricalTicks(
    1,
    contract,
    '20200529 13:15:00',
    '',
    1000,
    "TRADES",
    1,
    True,
    [])

# Sleep to allow enough time for data to be returned:
time.sleep(5)

# Working with Pandas DataFrames:
df = pd.DataFrame(
    app.data,
    columns=
    ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Converting timestamp to date and time:
df['DateTime'] = pd.to_datetime(df['DateTime'], unit='s')

df1 = pd.DataFrame(
    app.tick,
    columns=
    ['Time', 'Price', 'Size'])

app.disconnect()

tm = []
for i in df1['Time']:
    tm.append(_convert_time(i))

df1['Time2'] = tm"""

import pandas as pd
import helper.fundamental_data as fd
from helper.ibapi_downloader import IBapi

reqId = 0
tickerID = 1
ticker = 'PETR4'
sectype = 'STK'
exchange = 'BOVESPA'
currency = 'BRL'
end_period = ''
interval = '12 M'
timePeriod = '1 day'
datatype = 'ADJUSTED_LAST'
quarterly = True
ascending = True
separated = True

# Downloading data:
ibapi = IBapi()
df = ibapi.get_hist_fundamental_data(reqId, ticker, sectype, exchange,
                                     currency)

df = ibapi.get_hist_time_data(
    tickerID, ticker, sectype, exchange, currency, end_period, interval,
    timePeriod, datatype)

# Downloading income statement:
income_statement = fd.get_income_statement(
    ticker, quarterly, ascending)

# Downloading financial statement:
financial_statement = fd.get_financial_statement(
    ticker, quarterly, ascending, separated)

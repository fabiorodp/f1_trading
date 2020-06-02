from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import threading
import time


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def historicalData(self, reqId, bar):
        # print(f'Time: {bar.date} Close: {bar.close}')
        app.data.append([bar.date, bar.open, bar.high, bar.low, bar.close,
                         bar.volume, bar.average])

    def historicalTicksLast(self, reqId, ticks, done):
        for tick in ticks:
            # print("HistoricalTickLast. ReqId:", reqId, tick)
            app.tick.append([tick.time, tick.price, tick.size])


def run_loop():
    app.run()


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
app.tick = []


app.reqHistoricalTicks(
    1,
    contract,
    '20200529 13:15:00',
    '',
    1000,
    "TRADES",
    1,
    True,
    []
)

# Sleep to allow enough time for data to be returned:
time.sleep(5)

# Working with Pandas DataFrames:
df1 = pd.DataFrame(
    app.tick,
    columns=
    ['Time', 'Price', 'Size']
)

app.disconnect()


def convert_time(time):
    from datetime import datetime
    ts = int(time)
    return datetime.utcfromtimestamp(ts).strftime('%Y%m%d %H:%M:%S')


tm = []
for i in df1['Time']:
    tm.append(convert_time(i))

df1['Time2'] = tm

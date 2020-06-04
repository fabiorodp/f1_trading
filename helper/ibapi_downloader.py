# -*- coding: utf-8 -*-

"""
Helper functions to download historical quotation data from IBapi.
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

api = EClient(EWrapper())
api.connect('127.0.0.1', 4001, 0)
# api_thread = threading.Thread(target=api.run, daemon=True)
# api_thread.start()
c = Contract()
c.symbol = 'PETR4'
c.secType = 'STK'
c.exchange = 'BOVESPA'
c.currency = 'BRL'
api.reqFundamentalData(0, c, 'ReportSnapshot', [])
api.disconnect()


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.ticks = []
        self.fundamental_data = []
        self.ReportsFinSummary = []
        self.ReportsOwnership = []
        self.ReportSnapshot = []
        self.ReportsFinStatements = []
        self.RESC = []
        self.CalendarReport = []

    def historicalData(self, reqId, bar):
        self.data.append([bar.date, bar.open, bar.high, bar.low,
                          bar.close, bar.volume])

    def historicalTicksLast(self, reqId, ticks, done):
        for tick in ticks:
            self.ticks.append([tick.time, tick.price, tick.size])

    def fundamentalData(self, reqId, data):
        self.data.append(data)

    def run_loop(self):
        self.run()

    def get_hist_time_data(self,
                           tickerID, ticker, sectype, exchange, currency,
                           end_period,
                           interval, timePeriod, datatype, RTH=0,
                           timeFormat=1,
                           streaming=False, ip='127.0.0.1', port=4001, id=0):
        self.connect(ip, port, id)

        # Start the socket in a thread:
        api_thread = threading.Thread(target=self.run_loop, daemon=True)
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

    def get_hist_tick_data(self,
                           tickerID, ticker, sectype, exchange, currency,
                           start_period, end_period, num_ticks, datatype,
                           RTH=0,
                           ip='127.0.0.1', port=4001, id=0):
        self.connect(ip, port, id)

        # Start the socket in a thread:
        api_thread = threading.Thread(target=self.run_loop, daemon=True)
        api_thread.start()

        time.sleep(1)

        # Creating contract object:
        c = Contract()
        c.symbol = ticker
        c.secType = sectype
        c.exchange = exchange
        c.currency = currency

        self.reqHistoricalTicks(
            tickerID,  # Ticker ID
            c,  # Contract
            start_period,  # start period
            end_period,  # end period
            num_ticks,  # number of ticks
            datatype,  # 'TRADES' - Data Type  # ADJUSTED_LAST
            RTH,  # regular trading hours (1), or all available hours (0)
            True,  # ignoreSize
            [])

        time.sleep(5)

        df = pd.DataFrame(self.ticks, columns=['Time', 'Price', 'Size'])

        self.disconnect()

        return df

    def get_hist_fundamental_data(self, reqId, ticker, sectype, exchange,
                                  currency, ip='127.0.0.1', port=4001, id=0):
        """
        Parameters
        ----------
        tickerID
        ticker
        sectype
        exchange
        currency
        start_period
        end_period
        num_ticks
        datatype
        RTH
        ip
        port
        id

        Returns
        -------

        Notes
        -------
        Available reports:

        'ReportsFinSummary':	    Financial summary
        'ReportsOwnership':	        Company's ownership
        'ReportSnapshot':	        Company's financial overview
        'ReportsFinStatements':	    Financial Statements
        'RESC':	                    Analyst Estimates
        'CalendarReport':	        Company's calendar
        """
        self.connect(ip, port, id)

        # Start the socket in a thread:
        api_thread = threading.Thread(target=self.run_loop, daemon=True)
        api_thread.start()

        time.sleep(1)

        # Creating contract object:
        c = Contract()
        c.symbol = ticker
        c.secType = sectype
        c.exchange = exchange
        c.currency = currency

        """content = {
            'ReportsFinSummary': [], 'ReportsOwnership': [],
            'ReportSnapshot': [], 'ReportsFinStatements': [],
            'RESC': [], 'CalendarReport': []
        }"""

        report_types = ['ReportsFinSummary', 'ReportsOwnership',
                        'ReportSnapshot', 'ReportsFinStatements',
                        'RESC', 'CalendarReport']

        t = 'ReportsFinSummary'
        self.reqFundamentalData(
            reqId,  # int:: request's unique identifier
            c,  # object:: Contract,
            t,  # string:: report type
            []
        )

        time.sleep(5)

        # df = pd.DataFrame(self.ticks, columns=['Time', 'Price', 'Size'])
        df = self.data

        self.disconnect()

        return df

"""
# from ibapi.opt import ibConnection
from time import sleep
import csv

class Downloader(object):
    tickType47value = ''
    #field4price = ''

    def __init__(self):
        self.tws = ibConnection('localhost', 7496, 20)
        self.tws.register(self.tickPriceHandler, 'TickString')
        self.tws.connect()
        self._reqId = 1003 # current request id

    def tickPriceHandler(self,msg):
        if msg.tickType == 47:    # tickType=47
            self.tickType47value = msg.value
            #print('[debug]', msg)

    def requestData(self,contract):
        self.tws.reqMktData(self._reqId, contract, "233, 236, 258", False)  #"233, 236, 258",
        self._reqId+=1

    def cancelData(self):
        #self.tws.cancelMktData(1003)
        self.tws.disconnect()


if __name__=='__main__':
    headers = ['TickNo',
            'TTMNPMGN',
             'NLOW',
             'TTMPRCFPS',
             'TTMGROSMGN',
             'TTMCFSHR',
             'QCURRATIO',
             'TTMREV',
             'TTMINVTURN',
             'TTMOPMGN',
             'TTMPR2REV',
             'AEPSNORM',
             'TTMNIPEREM',
             'EPSCHNGYR',
             'TTMPRFCFPS',
             'TTMRECTURN',
             'TTMPTMGN',
             'QCSHPS',
             'TTMFCF',
             'LATESTADATE',
             'APTMGNPCT',
             'AEBTNORM',
             'TTMNIAC',
             'NetDebt_I',
             'PRYTDPCTR',
             'TTMEBITD',
             'AFEEPSNTM',
             'PR2TANBK',
             'EPSTRENDGR',
             'QTOTD2EQ',
             'TTMFCFSHR',
             'QBVPS',
             'NPRICE',
             'YLD5YAVG',
             'REVTRENDGR',
             'TTMEPSXCLX',
             'QTANBVPS',
             'PRICE2BK',
             'MKTCAP',
             'TTMPAYRAT',
             'TTMINTCOV',
             'TTMREVCHG',
             'TTMROAPCT',
             'TTMROEPCT',
             'TTMREVPERE',
             'APENORM',
             'TTMROIPCT',
             'REVCHNGYR',
             'CURRENCY',
             'DIVGRPCT',
             'TTMEPSCHG',
             'PEEXCLXOR',
             'QQUICKRATI',
             'TTMREVPS',
             'BETA',
             'TTMEBT',
             'ADIV5YAVG',
             'ANIACNORM',
             'QLTD2EQ',
             'NHIG']

    stocks=['700']
    with open('Your path', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"')
        csvwriter.writerow(headers)
        sleep(0.5)

    for x in stocks:
        for _ in range(5): #10 #If markets are open, there can be no more than 5 requests pending for the same contract.
            dl = Downloader()
            c = Contract()
            c.m_symbol = x
            c.m_secType = 'STK'
            c.m_exchange = 'SEHK'
            c.m_currency = 'HKD'
            sleep(1)
            dl.requestData(c)
            sleep(1)
            m0 = str(x)
            m = dl.tickType47value
            #data = m.split(';')
            #pairs = { tuple(datum.split('=')) for datum in data}
            #print(m)
            sleep(1)

            if dl.tickType47value:
                    with         open(r'c:\\Users\\Owner\\Desktop\\extracedCSV\\ALLHKSTOCK@Finratio_2.csv', 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"')
                    row = []
                    row.append(m0)
                    row.append(m)
                    csvwriter.writerow(row)
                    dl.cancelData()
                    sleep(0.5)
                    break

            print("Data is empty")
            dl.cancelData()
            sleep(0.5)"""

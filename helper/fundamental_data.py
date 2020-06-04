# -*- coding: utf-8 -*-

"""
Helper functions to download historical financial and income statement data
of companies listed at B3 (BMF/BOVESPA).
"""

__author__ = "Fábio Rodrigues Pereira"
__email__ = "fabio@fabiorodriguespereira.com"

import io
import os
import xlrd
import requests
import pandas as pd
from zipfile import ZipFile


def _convert_type(x):
    if str(x).endswith('%'):
        return float(x[:-1].replace('.', '').replace(',', '.')) / 100
    try:
        if float(x).is_integer():
            return int(float(x))
        return float(x)
    except ValueError:
        return None


def b3_tickers():
    """
    Downloads tickers' codes from Fundamentus website

    Returns
    -------
    df : pd.DataFrame
        Pandas' DataFrame with three columns:
            'Papel':            Ticker
            'Nome Comercial':   Trade Name
            'Razão Social':     Corporate Name
    """
    # Requesting the html code:
    html = requests.get('http://fundamentus.com.br/detalhes.php').text

    # Getting the information wanted:
    df = pd.read_html(html)[0]

    return df


def _get_data(ticker, quarterly, ascending):
    """
    Download zip file with excel file containing the historical financial
    data of a specific stock ticker.

    Parameters
    ----------
    ticker : string
        The company's ticker.

    quarterly : bool
        True for getting quarterly historical data, False for not.

    ascending : bool
        True for getting ascending historical data, False for descending.

    Returns
    -------
    dfs : dict
        Dictionary with Pandas' DataFrame as values of the following keys:
            \'Bal. Patrim.\': financial statement
            \'Dem. Result.\': income statement
    """
    # Making string with upper case:
    ticker = ticker.upper()

    # Getting html:
    r = requests.get(
        'https://www.fundamentus.com.br/balancos.php',
        params={'papel': ticker})

    # Getting SID of the file:
    SID = r.cookies.values()[0]

    # Downloading the file:
    response_sheet = requests.get(
        'https://www.fundamentus.com.br/planilhas.php',
        params={'SID': SID})

    # Error if file not found:
    if response_sheet.text.startswith('Ativo nao encontrado'):
        raise IndexError(f'Couldn\'t find any data for {ticker}')

    # Decompressing file:
    with io.BytesIO(response_sheet.content) as zip_bytes:
        with ZipFile(zip_bytes) as zip:
            xls = zip.read('balanco.xls')

    # Accessing data in the 2 different excel sheets:
    wb = xlrd.open_workbook(file_contents=xls,
                            logfile=open(os.devnull, 'w'))

    # Saving contents as DataFrame in a dictionary:
    dfs = {
        'Bal. Patrim.': pd.read_excel(wb, engine='xlrd',
                                      index_col=0,
                                      sheet_name='Bal. Patrim.'),

        'Dem. Result.': pd.read_excel(wb, engine='xlrd',
                                      index_col=0,
                                      sheet_name='Dem. Result.')
    }

    for sheet, df in dfs.items():
        # Cleaning the DataFrame
        df.columns = df.iloc[0, :]
        df = df.iloc[1:].T.applymap(_convert_type)
        df.index.name = 'Data'
        df.columns.name = ticker
        df = df.loc[df.index.notnull()]
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y')

        if not quarterly:
            rows_to_drop = [x for x in df.index.year
                            if list(df.index.year).count(x) != 4]
            df = df.groupby(df.index.year).sum()
            df.drop(rows_to_drop, inplace=True)
            df.index = [str(x) for x in df.index]

        dfs[sheet] = df.sort_index(ascending=ascending).astype(int)

    return dfs


def get_financial_statement(ticker, quarterly=False, ascending=True,
                            separated=True):
    """
    Get the financial statement as Pandas' DataFrame.

    Parameters
    ----------
    ticker : string
        The company's ticker.

    quarterly : bool
        True for getting quarterly historical data, False for not.

    ascending : bool
        True for getting ascending historical data, False for descending.

    separated : bool
        If True, the DataFrame will be hierarchically divided by super
        columns:
            \'Ativo Total\':                Total Assets
            \'Ativo Circulante\':           Current Assets
            \'Ativo Não Circulante\':       Non-current Assets
            \'Passivo Total\':              Total Liabilities
            \'Passivo Circulante\':         Current Liabilities
            \'Passivo Não Circulante\':     Non-current Liabilities
            \'Patrimônio Líquido\':         Net Worth

    Returns
    -------
    df : pd.DataFrame
        Containing all the historical financial statement.

    Notes
    ----------
    Highly recommended to use True for the parameter 'separated' because
    some infra columns are duplicated which could lead to confusion.
    """
    # Downloading data:
    df = _get_data(ticker, quarterly=quarterly,
                   ascending=ascending)['Bal. Patrim.']

    if separated:
        super_cols = [
            'Ativo Total',
            'Ativo Circulante',
            'Ativo Não Circulante',
            'Ativo Realizável a Longo Prazo',
            'Passivo Total',
            'Passivo Circulante',
            'Passivo Não Circulante',
            'Passivo Exigível a Longo Prazo',
            'Patrimônio Líquido'
        ]

        cols = list(df.columns)

        # Handling different balance sheets for banks and other companies
        if 'Ativo Não Circulante' in cols:
            super_cols.remove('Ativo Realizável a Longo Prazo')
        else:
            super_cols.remove('Ativo Não Circulante')

        if 'Passivo Não Circulante' in cols:
            super_cols.remove('Passivo Exigível a Longo Prazo')
        else:
            super_cols.remove('Passivo Não Circulante')

        idxs = [cols.index(x) for x in cols if x in super_cols]

        slices = [slice(idxs[i], idxs[i + 1])
                  if i < (len(idxs) - 1)
                  else slice(idxs[i], None)
                  for i, _ in enumerate(idxs)]

        tuples = []

        for s in slices:
            sup = super_cols.pop(0)

            # Renaming columns to standardize different companies' DataFrame
            if sup == 'Ativo Realizável a Longo Prazo':
                sup = 'Ativo Não Circulante'
            if sup == 'Passivo Exigível a Longo Prazo':
                sup = 'Passivo Não Circulante'

            for col in cols[s]:
                tuples.append((sup, col))

        df.columns = pd.MultiIndex.from_tuples(tuples)

    return df


def get_income_statement(ticker, quarterly=False, ascending=True):
    """
    Get the income statement as Pandas' DataFrame.

    Parameters
    ----------
    ticker : string
        The company's ticker.

    quarterly : bool
        True for getting quarterly historical data, False for not.

    ascending : bool
        True for getting ascending historical data, False for descending.

    Returns
    -------
    df : pd.DataFrame
        Containing all the historical income statement.
    """
    df = _get_data(ticker, quarterly=quarterly,
                   ascending=ascending)['Dem. Result.']

    return df

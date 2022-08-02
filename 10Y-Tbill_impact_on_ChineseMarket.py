'''
Author: Xin Yang
Date: June 13 2022
Part3 Pickup the useful factor：
As we said before:
The Whole model includes 3 things.
1. Select the factor: It needs genius's intuition in finance and investment, so experience and finance knowledge is important here
2. Get the exposure: The data cleaning will be the first step. Then regression could help us to do so.
3. Predict the return: After we make the prediction on factors, then we could make the prediction on the return.

The ability to capture a good factor from numerous factor is not easy indeed, it need skill from financial intuition.

The dropdown in China Market from Dec. 2021 - May 2022 is severe, the reason is still not clear.

My guess is that the 10Y-Tbill's interest rate's increase leads to the dropdown. We may check it in this document.

API : AKShare
'''
import os
import statsmodels.api as sm
import scipy.optimize as sp
import numpy as np
import akshare as ak
import pandas as pd

def main(frequency: str, benchmark: str, startdate: str, enddate: str) -> pd.DataFrame:
    '''
    This method is to do get the data of 10Y-Till and do the regression, to see its' impact on 上证指数.
    :param period: weekly
    :param benchmark: SZZS 000001
    :param startdate: 2021.12
    :param enddate: 2022.5
    :return: stocks' beta -> pd.Dataframe
    '''
    output = dict()

    "First step is for get the raw data "
    bond_zh_us_rate_df = ak.bond_zh_us_rate()
    bond_zh_us_rate_df['日期'] = bond_zh_us_rate_df['日期'].astype('str')
    bond_zh_us_rate_df = bond_zh_us_rate_df.set_index('日期')
    bond_zh_us_rate_df = bond_zh_us_rate_df['美国国债收益率10年']
    #print(bond_zh_us_rate_df)

    benchmark_return = ak.index_zh_a_hist(symbol=benchmark, period=frequency, start_date=startdate, end_date=enddate)
    benchmark_return = benchmark_return.set_index('日期')
    #enchmark_return = benchmark_return['涨跌幅']

    benchmark_return = benchmark_return.join(bond_zh_us_rate_df,how='left')
    table = data_cleaning_remove_NaN(benchmark_return)[['涨跌幅','美国国债收益率10年']]

    Normalized = table['美国国债收益率10年'].tolist()
    Normalized2 = table['美国国债收益率10年'].tolist()
    for each in range(1, len(Normalized)):
        Normalized[each] = Normalized2[each] - Normalized2[each-1]

    table['美国国债收益率10年'] = Normalized
    table = table[1:]

    Y_array = table['涨跌幅'].to_numpy()
    X_array = table['美国国债收益率10年'].to_numpy()
    result = sm.OLS(Y_array, X_array).fit()
    print(result.summary())
def data_cleaning_remove_NaN(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This method is to remove the NaN value due to the joining function of dataframe
    :param data: data need for cleaning
    :return: the dataframe after removing the NaN value
    '''
    return data.query('美国国债收益率10年 >= 0')
main("weekly",'000001','20211120','20220501')



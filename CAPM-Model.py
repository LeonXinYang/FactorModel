'''
Author: Xin Yang
Date: June 12 2022
Part1 CAPM Model：
In this package, we may try to re-build the classic CAPM, Fama-French 3 Factors' model.
If available, we also may try to build up our own factor and check their validity

Due to the running time and space, in this memo model, we may use the 上证50 stocks as our range.

The Whole model includes 3 things.
1. Select the factor: It needs genius's intuition in finance and investment, so experience and finance knowledge is important here
2. Get the exposure: The data cleaning will be the first step. Then regression could help us to do so.
3. Predict the return: After we make the prediction on factors, then we could make the prediction on the return.

API : AKShare
'''
import math
import os
import statsmodels.api as sm
import scipy.optimize as sp
import numpy as np
import akshare as ak
import pandas as pd

'''
Part 1: CAPM Model
'''

def CAPM_model_residual(beta, y, x, rf) -> float:
    '''
    This is the loss function of CAPM, the same calculation to the OLS algorithm, in order to get the appropriate beta
    :param beta: dependent_variable
    :param y: shown as training samples
    :param x: shown as training samples
    :param rf: shown as no-risk bond return rate
    :return: value of loss
    '''
    return sum(np.array((y - beta * (x - rf) - rf)**2))


def CAPM_model(stocks_group: str, frequency: str, benchmark: str, startdate: str, enddate: str) -> pd.DataFrame:
    '''
    This method is to do the regression and get the beta of CAPM for each stock using the historical record.
    In here, as an example, the parameter will be the following
    :param stocks_group: SZ50 000016
    :param period: 2019-2020
    :param benchmark: SZZS 000001
    :param startdate: 20190101
    :param enddate: 20200101
    :return: stocks' beta -> pd.Dataframe
    '''
    output = dict()

    "First step is for get the raw data "
    index_stock_cons_df = ak.index_stock_cons(symbol=stocks_group)
    #print(index_stock_cons_df)

    index_zh_a_hist_df = ak.index_zh_a_hist(symbol=benchmark, period=frequency, start_date=startdate, end_date=enddate)
    index_zh_a_hist_df = data_cleaning(index_zh_a_hist_df)
    #print(index_zh_a_hist_df)

    "The no RISK return rate，we select 5-year chinese national bond as Risk-free return rate"
    bond_zh_us_rate_df = ak.bond_zh_us_rate()
    bond_zh_us_rate_df['日期'] = bond_zh_us_rate_df['日期'].astype('str')
    stock_num = index_stock_cons_df['品种代码'].to_list()

    "Second step is to calculate the beta for each individual stock"
    beta_group = []
    stocks_group_output = []
    name_output = []
    r2 = []
    p_value = []
    num = -1
    for each in stock_num:
        num += 1
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=each, period=frequency, start_date=startdate, end_date=enddate, adjust="qfq")
        #print(stock_zh_a_hist_df)
        if len(stock_zh_a_hist_df.keys()) == 0:
            continue
        else:
            stock_zh_a_hist_df = stock_zh_a_hist_df[['日期','涨跌幅']]

            joined_table = stock_zh_a_hist_df.set_index('日期').join(index_zh_a_hist_df.set_index('日期'),how='left',lsuffix='a',rsuffix='b')
            joined_table = joined_table.join(bond_zh_us_rate_df.set_index('日期')['中国国债收益率5年'],how='left',rsuffix='c')
            joined_table = data_cleaning_remove_NaN(joined_table)
            beta0 = 1
            beta_output = sp.fmin(CAPM_model_residual, beta0,args=(np.array(joined_table['涨跌幅a'].to_list()),np.array(joined_table['涨跌幅b'].to_list()),np.array(joined_table['中国国债收益率5年']/52).tolist()))
            print(beta_output)

            result = sm.OLS((np.array(joined_table['涨跌幅a'])-np.array(joined_table['中国国债收益率5年']/52)),np.array(joined_table['涨跌幅b'])-np.array(joined_table['中国国债收益率5年']/52)).fit()
            print(result.summary())
            r2.append(result.rsquared)
            p_value.append(result.pvalues)
            beta_group.append(sum(beta_output))
            stocks_group_output.append(each)
            name_output.append(index_stock_cons_df['品种名称'][num])
    output['品种名称'] = name_output
    output['代码'] = stocks_group_output
    output['Beta'] = beta_group
    output['R^2'] = r2
    output['p_value'] = p_value
    df_output = pd.DataFrame(output)
    if not os.listdir().__contains__('CAPM_output'):
        os.makedirs("CAPM_output")
    df_output.to_csv("CAPM_output/" + stocks_group + "-" + frequency + "-" + startdate + "-" + enddate + "-" + '.csv')
    print(df_output)
    return df_output

def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This method is for removing the unused data for improving running speed
    :param data: data need for cleaning, by reducing the useless dimension
    :return: the data after cleaned
    '''
    return data[['日期','涨跌幅']]

def data_cleaning_remove_NaN(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This method is to remove the NaN value due to the joining function of dataframe
    :param data: data need for cleaning
    :return: the dataframe after removing the NaN value
    '''
    return data.query('涨跌幅b >= 0 or 涨跌幅b <= 0')

CAPM_model('000016', 'weekly', '000001', '20190101', '20200101')

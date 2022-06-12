'''
Author: Xin Yang
Date: June 12 2022
Part2 Fama-French-3-Factors Model：
In this package, we may try to re-build the classic Fama-French 3 Factors' model.
If available, we also may try to build up our own factor and check their validity

Due to the running time and space, in this memo model, we may use the 上证50 stocks as our range.

The Whole model includes 3 things.
1. Select the factor: It needs genius's intuition in finance and investment, so experience and finance is important here
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
Part 2: Fama-French-3-Factors Model
'''
def build_SMB_factor(startdate: str, enddate: str) -> pd.DataFrame:
    '''
    This method is for getting the SMB Factor value.
    !!! However, Liu et al. (2018) points out that the SMB does not work in China based on the "SHELL-Value Contamination"
    In here, we remove the 30% of the smallest value company before calculating the SMB factor.

    Liu, J., R. F. Stambaugh, and Y. Yuan (2018). Size and Value in China. Journal of Financial Economics, forthcoming.

    :param startdate: startdate of the calculation
    :param enddate: enddate of the calculation
    :return: the dataframe with calendar as index while with SMB factor as output
    '''
    stock_sh_a_spot_em_df = ak.stock_sh_a_spot_em()
    stock_sh_a_spot_em_df1 = stock_sh_a_spot_em_df.sort_values(by=["总市值"],ascending = [False])

    #Remove the last 1/3 of the stock in our calculation due to the shell value contamination
    stock_sh_a_spot_em_df2 = stock_sh_a_spot_em_df1[:int(len(stock_sh_a_spot_em_df1)*2/3)]
    stock_sh_a_spot_em_df_big = stock_sh_a_spot_em_df2[:int(len(stock_sh_a_spot_em_df2)/2)]
    stock_sh_a_spot_em_df_small = stock_sh_a_spot_em_df2[int(len(stock_sh_a_spot_em_df2)/2):]

    #Normalisation
    stock_sh_a_spot_em_df_big = stock_sh_a_spot_em_df_big.set_index('代码')
    weight_big = stock_sh_a_spot_em_df_big['总市值'] / sum(stock_sh_a_spot_em_df_big['总市值'])

    stock_sh_a_spot_em_df_small = stock_sh_a_spot_em_df_small.set_index('代码')
    weight_small = stock_sh_a_spot_em_df_small['总市值'] / sum(stock_sh_a_spot_em_df_small['总市值'])

    def helper_function(weight_big: pd.DataFrame, stock_sh_a_spot_em_df_big: pd.DataFrame,startdate: str, enddate: str) -> pd.DataFrame:
        '''
        Helper function to iterate each stock, calculate their return rate under the weight
        :param weight_big: the weight dataframe
        :param stock_sh_a_spot_em_df_big: the element of stocks which need iteration
        :param startdate: startdate
        :param enddate: enddate
        :return: the output as Big or Small Stock's return rate after giving the weight
        '''
        output = None
        n = 0
        f_output = pd.DataFrame()
        for each in stock_sh_a_spot_em_df_big.index.tolist():
            try:
                stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=each, period="weekly", start_date=startdate, end_date=enddate, adjust="qfq")[['日期','涨跌幅']]
                stock_zh_a_hist_df['涨跌幅'] *= weight_big[each]
                if output is None:
                    output = stock_zh_a_hist_df
                    output = output.set_index('日期')
                else:
                    output = output.join(stock_zh_a_hist_df.set_index('日期'),how='left',rsuffix=str(n))
                    if n % 100 == 0:
                        print(output)
            except:
                continue
            n+=1
        output = output.fillna(value = 0)
        f_output['sum'] = output.sum(axis=1)
        #f_output['日期'] = output.index
        return f_output
    big = helper_function(weight_big,stock_sh_a_spot_em_df_big,startdate,enddate)
    small = helper_function(weight_small,stock_sh_a_spot_em_df_small,startdate,enddate)
    if not os.listdir().__contains__('SMB'):
        os.makedirs('SMB')
    big.to_csv('SMB/big.csv')
    small.to_csv('SMB/small.csv')
    big['sum'] = big['sum'] - small['sum']
    #big.set_index('日期')
    big.to_csv('SMB/market_Value_factor.csv')
    return big

#build_SMB_factor('20190101', '20200101')


def build_HML_factor(startdate: str, enddate: str) -> pd.DataFrame:
    '''
    This method is for getting the HML Factor value.
    !!! However, Liu et al. (2018) points out that the HML does not work in China based on the "SHELL-Value Contamination"
    In here, we remove the 30% of the smallest value company before calculating the HML factor.

    Liu, J., R. F. Stambaugh, and Y. Yuan (2018). Size and Value in China. Journal of Financial Economics, forthcoming.

    :param startdate: startdate of the calculation
    :param enddate: enddate of the calculation
    :return: the dataframe with calendar as index while with HML factor as output
    '''

    stock_sh_a_spot_em_df = ak.stock_sh_a_spot_em()
    stock_sh_a_spot_em_df1 = stock_sh_a_spot_em_df.sort_values(by=["总市值"],ascending = [False])

    #Remove the last 1/3 of the stock in our calculation due to the shell value contamination
    stock_sh_a_spot_em_df2 = stock_sh_a_spot_em_df1[:int(len(stock_sh_a_spot_em_df1)*2/3)]
    stock_sh_a_spot_em_df2 = stock_sh_a_spot_em_df2.sort_values(by=["市净率"],ascending = [False])[stock_sh_a_spot_em_df.市净率 > 0]
    stock_sh_a_spot_em_df_big = stock_sh_a_spot_em_df2[:int(len(stock_sh_a_spot_em_df2)/2)]
    stock_sh_a_spot_em_df_small = stock_sh_a_spot_em_df2[int(len(stock_sh_a_spot_em_df2)/2):]

    #Normalisation
    stock_sh_a_spot_em_df_big = stock_sh_a_spot_em_df_big.set_index('代码')
    weight_big = stock_sh_a_spot_em_df_big['总市值'] / sum(stock_sh_a_spot_em_df_big['总市值'])

    stock_sh_a_spot_em_df_small = stock_sh_a_spot_em_df_small.set_index('代码')
    weight_small = stock_sh_a_spot_em_df_small['总市值'] / sum(stock_sh_a_spot_em_df_small['总市值'])

    def helper_function(weight_big: pd.DataFrame, stock_sh_a_spot_em_df_big: pd.DataFrame,startdate: str, enddate: str) -> pd.DataFrame:
        '''
        Helper function to iterate each stock, calculate their return rate under the weight
        :param weight_big: the weight dataframe
        :param stock_sh_a_spot_em_df_big: the element of stocks which need iteration
        :param startdate: startdate
        :param enddate: enddate
        :return: the output as Big or Small P/B Stock's return rate after giving the weight
        '''
        output = None
        n = 0
        f_output = pd.DataFrame()
        for each in stock_sh_a_spot_em_df_big.index.tolist()[1:]:
            try:
                stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=each, period="weekly", start_date=startdate, end_date=enddate, adjust="qfq")[['日期','涨跌幅']]
                stock_zh_a_hist_df['涨跌幅'] *= weight_big[each]
                if output is None:
                    output = stock_zh_a_hist_df
                    output = output.set_index('日期')
                else:
                    output = output.join(stock_zh_a_hist_df.set_index('日期'),how='left',rsuffix=str(n))
                    if n % 100 == 0:
                        print(output)
            except:
                continue
            n+=1
        output = output.fillna(value = 0)
        f_output['sum'] = output.sum(axis=1)
        #f_output['日期'] = output.index
        return f_output
    big = helper_function(weight_big,stock_sh_a_spot_em_df_big,startdate,enddate)
    small = helper_function(weight_small,stock_sh_a_spot_em_df_small,startdate,enddate)
    if not os.listdir().__contains__('HML'):
        os.makedirs('HML')
    big.to_csv('HML/bigPB.csv')
    small.to_csv('HML/smallPB.csv')
    big['sum'] = big['sum'] - small['sum']
    #big.set_index('日期')
    big.to_csv('HML/PB_factor.csv')
    return big

#build_HML_factor('20190101', '20200101')

def Fama_French_3_model(stocks_group: str, frequency: str, benchmark: str, startdate: str, enddate: str) -> pd.DataFrame:
    '''
    This method is to do the regression and get the beta of Fama-French 3 factors' model for each stock using the historical record.
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
    f_p_value = []
    cons = []
    p_value_cons = []
    market = []
    p_value_Market = []
    smb = []
    p_value_SMB = []
    hml = []
    p_value_HML = []
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
            SMB = pd.read_csv("SMB/market_Value_factor.csv")
            HML = pd.read_csv('HML/PB_factor.csv')
            joined_table = joined_table.join(SMB.set_index('日期')['sum'],how='left',rsuffix='c')
            joined_table = joined_table.join(HML.set_index('日期')['sum'],how='left',rsuffix='c')
            joined_table['中国国债收益率5年'] *= 1 / 52
            joined_table['涨跌幅b'] -= joined_table['中国国债收益率5年']
            joined_table = data_cleaning_remove_NaN(joined_table)
            if not os.listdir().__contains__('3F_data'):
                os.makedirs('3F_data')
            joined_table.to_csv('3F_data/Fama3-'+startdate+"-"+enddate+"-"+".csv")
            X_array = joined_table[['涨跌幅b','sum','sumc']].to_numpy()
            X_array = sm.add_constant(X_array)
            result = sm.OLS((np.array(joined_table['涨跌幅a'])-np.array(joined_table['中国国债收益率5年']/52)), X_array).fit()
            print(result.summary())
            #print(result.params)

            r2.append(result.rsquared)
            f_p_value.append(result.f_pvalue)
            p_value_cons.append(result.pvalues[0])
            p_value_Market.append(result.pvalues[1])
            p_value_SMB.append(result.pvalues[2])
            p_value_HML.append(result.pvalues[3])
            cons.append(result.params[0])
            market.append(result.pvalues[1])
            smb.append(result.pvalues[2])
            hml.append(result.pvalues[3])
            stocks_group_output.append(each)

            name_output.append(index_stock_cons_df['品种名称'][num])
    output['品种名称'] = name_output
    output['代码'] = stocks_group_output
    output['R^2'] = r2
    output['f_p_value'] = f_p_value
    output['coeff_constant'] = cons
    output['p_value_constant'] = p_value_cons
    output['coeff_market'] = market
    output['p_value_Market'] = p_value_Market
    output['coeff_SMB'] = smb
    output['p_value_SMB'] = p_value_SMB
    output['coeff_HML'] = hml
    output['p_value_HML'] = p_value_HML

    df_output = pd.DataFrame(output)
    if not os.listdir().__contains__('3F_output'):
        os.makedirs("3F_output")
    df_output.to_csv("3F_output/" + stocks_group + "-" + frequency + "-" + startdate + "-" + enddate + "-" + '.csv')
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

Fama_French_3_model('000016', 'weekly', '000001', '20190101', '20200101')

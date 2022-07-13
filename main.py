import yfinance as yf
import datetime as dt
import pandas
import ta
import copy
import tsfresh
import numpy as np
import xgboost
import tqdm

days = 7

def run_tsfresh(df, y):
    df, y = copy.copy((df, y))
    df['id'] = 1
    df['t'] = range(len(df))
    df = tsfresh.extract_relevant_features(df, y,
        default_fc_parameters=tsfresh.feature_extraction.EfficientFCParameters(),
        column_id='id', column_sort='t')
    
    return tsfresh.utilities.dataframe_functions.impute_dataframe_zero(df)

if __name__ == '__main__':
    ticker = yf.Ticker(input(''))
    df = ticker.history(period='10y', interval='1d')
    df = df.sort_index()
    
    add = lambda date, days: date + dt.timedelta(days=days)
    #df = df[:add(df.index[-1], -days)]
    true_lo = pandas.Series([
        min(df.loc[df.index[i]:add(df.index[i], days),'Low'])
        for i in range(len(df))
    ])
    true_lo.index = df.index
    
    true_hi = pandas.Series([
        max(df.loc[df.index[i]:add(df.index[i], days),'High'])
        for i in range(len(df))
    ])
    true_hi.index = df.index
    
    df = ta.add_all_ta_features(df,
        open='Open', high='High', low='Low', close='Close', volume='Volume', 
    fillna=True)
    df['date1'] = list(map(lambda row: row.month, df.index))
    df['date2'] = list(map(lambda row: row.day, df.index))
    df['date3'] = list(map(lambda row: row.dayofweek, df.index))
    
    test_set = np.random.permutation(len(df) // days)
    test_set = test_set[:round(.25 * len(test_set))]
    test_set_a = test_set[:round(.5 * len(test_set))]
    test_set_a = [(i * days) + j for j in range(days) for i in test_set_a]
    test_set_a = np.array([i in test_set_a for i in range(len(df))])
    test_set_b = test_set[round(.5 * len(test_set)):]
    test_set_b = [(i * days) + j for j in range(days) for i in test_set_b]
    test_set = [(i * days) + j for j in range(days) for i in test_set]
    test_set = np.array([i in test_set for i in range(len(df))])
    
    soln_lo = xgboost.XGBRegressor(
        verbosity=2,
        tree_method='hist'
    )
    soln_lo.fit(df[~test_set], true_lo[~test_set])
    
    soln_hi = xgboost.XGBRegressor(
        verbosity=2,
        tree_method='hist'
    )
    soln_hi.fit(df[~test_set], true_hi[~test_set])
    
    lo = soln_lo.predict(df[test_set_a])
    dif_lo = lo - true_lo[test_set_a]
    avg_dif_lo = np.mean(dif_lo)
    sd_dif_lo = np.std(dif_lo)
    
    hi = soln_hi.predict(df[test_set_a])
    dif_hi = hi - true_hi[test_set_a]
    avg_dif_hi = np.mean(dif_hi)
    sd_dif_hi = np.std(dif_hi)
    
    test_lo = 0
    test_hi = 0 
    for i in tqdm.tqdm(test_set_b):
        lo, = soln_lo.predict(df.iloc[[i]])
        lo += avg_dif_lo - 2 * sd_dif_lo
        
        hi, = soln_hi.predict(df.iloc[[i]])
        hi += avg_dif_hi + 2 * sd_dif_hi
        
        test_lo += np.mean(df.loc[df.index[i]:add(df.index[i], days),'Low'] >= lo)
        test_hi += np.mean(df.loc[df.index[i]:add(df.index[i], days),'High'] <= hi)
    test_lo /= len(test_set_b)
    print(test_lo)
    test_hi /= len(test_set_b)
    print(test_hi)
    
    lo, = soln_lo.predict(df.iloc[[-1]])
    lo += avg_dif_lo - 2 * sd_dif_lo
    print('${:,.2f}'.format(lo))
    
    hi, = soln_hi.predict(df.iloc[[-1]])
    hi += avg_dif_hi + 2 * sd_dif_lo
    print('${:,.2f}'.format(hi))
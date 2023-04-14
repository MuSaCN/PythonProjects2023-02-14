# Author:Zhang Yuan
from MyPackage import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# ------------------------------------------------------------
__mypath__ = MyPath.MyClass_Path("")  # 路径类
mylogging = MyDefault.MyClass_Default_Logging(activate=False)  # 日志记录类，需要放在上面才行
myfile = MyFile.MyClass_File()  # 文件操作类
myword = MyFile.MyClass_Word()  # word生成类
myexcel = MyFile.MyClass_Excel()  # excel生成类
myini = MyFile.MyClass_INI()  # ini文件操作类
mytime = MyTime.MyClass_Time()  # 时间类
myparallel = MyTools.MyClass_ParallelCal()  # 并行运算类
myplt = MyPlot.MyClass_Plot()  # 直接绘图类(单个图窗)
mypltpro = MyPlot.MyClass_PlotPro()  # Plot高级图系列
myfig = MyPlot.MyClass_Figure(AddFigure=False)  # 对象式绘图类(可多个图窗)
myfigpro = MyPlot.MyClass_FigurePro(AddFigure=False)  # Figure高级图系列
myplthtml = MyPlot.MyClass_PlotHTML()  # 画可以交互的html格式的图
mynp = MyArray.MyClass_NumPy()  # 多维数组类(整合Numpy)
mypd = MyArray.MyClass_Pandas()  # 矩阵数组类(整合Pandas)
mypdpro = MyArray.MyClass_PandasPro()  # 高级矩阵数组类
myDA = MyDataAnalysis.MyClass_DataAnalysis()  # 数据分析类
myDefault = MyDefault.MyClass_Default_Matplotlib()  # 画图恢复默认设置类
# myMql = MyMql.MyClass_MqlBackups() # Mql备份类
# myBaidu = MyWebCrawler.MyClass_BaiduPan() # Baidu网盘交互类
# myImage = MyImage.MyClass_ImageProcess()  # 图片处理类
myBT = MyBackTest.MyClass_BackTestEvent()  # 事件驱动型回测类
myBTV = MyBackTest.MyClass_BackTestVector()  # 向量型回测类
myML = MyMachineLearning.MyClass_MachineLearning()  # 机器学习综合类
mySQL = MyDataBase.MyClass_MySQL(connect=False)  # MySQL类
mySQLAPP = MyDataBase.MyClass_SQL_APPIntegration()  # 数据库应用整合
myWebQD = MyWebCrawler.MyClass_QuotesDownload(tushare=False)  # 金融行情下载类
myWebR = MyWebCrawler.MyClass_Requests()  # Requests爬虫类
myWebS = MyWebCrawler.MyClass_Selenium(openChrome=False)  # Selenium模拟浏览器类
myWebAPP = MyWebCrawler.MyClass_Web_APPIntegration()  # 爬虫整合应用类
myEmail = MyWebCrawler.MyClass_Email()  # 邮箱交互类
myReportA = MyQuant.MyClass_ReportAnalysis()  # 研报分析类
myFactorD = MyQuant.MyClass_Factor_Detection()  # 因子检测类
myKeras = MyDeepLearning.MyClass_tfKeras()  # tfKeras综合类
myTensor = MyDeepLearning.MyClass_TensorFlow()  # Tensorflow综合类
myMT5 = MyMql.MyClass_ConnectMT5(connect=False)  # Python链接MetaTrader5客户端类
myMT5Pro = MyMql.MyClass_ConnectMT5Pro(connect=False)  # Python链接MT5高级类
myMT5Indi = MyMql.MyClass_MT5Indicator()  # MT5指标Python版
myMT5Report = MyMT5Report.MyClass_StratTestReport(AddFigure=False)  # MT5策略报告类
myMT5Analy = MyMT5Analysis.MyClass_ForwardAnalysis()  # MT5分析类
myMT5Lots_Fix = MyMql.MyClass_Lots_FixedLever(connect=False)  # 固定杠杆仓位类
myMT5Lots_Dy = MyMql.MyClass_Lots_DyLever(connect=False)  # 浮动杠杆仓位类
myMT5run = MyMql.MyClass_RunningMT5()  # Python运行MT5
myMT5code = MyMql.MyClass_CodeMql5()  # Python生成MT5代码
myMoneyM = MyTrade.MyClass_MoneyManage()  # 资金管理类
myDefault.set_backend_default("Pycharm")  # Pycharm下需要plt.show()才显示图
# ------------------------------------------------------------
# Jupyter Notebook 控制台显示必须加上：%matplotlib inline ，弹出窗显示必须加上：%matplotlib auto
# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import datetime
import pandas_datareader as pdr
import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')

# import sktime
import statsmodels as sm
import matplotlib
import sklearn
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              ExtraTreesRegressor)

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import HuberRegressor


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
# from sktime.forecasting.all import (
#         Deseasonalizer, Detrender,
#         temporal_train_test_split,
#         mean_absolute_percentage_error as mape,
#         mean_squared_percentage_error as mspe,
#         mean_squared_error as mse,
#         ForecastingHorizon,
#         NaiveForecaster,
#         TransformedTargetForecaster,
#         PolynomialTrendForecaster
# )
# from sktime.forecasting.compose import make_reduction
# from sktime.forecasting.all import (
#         ForecastingGridSearchCV,
#         SlidingWindowSplitter,
#         MeanAbsolutePercentageError)
from statsmodels.tsa.api import seasonal_decompose, adfuller
# from sktime.performance_metrics.forecasting import(MeanAbsolutePercentageError,
#                                                    MeanSquaredError,
#                                                    MeanAbsoluteScaledError)
from datetime import timedelta
from datetime import date

# mse = MeanSquaredError()
# mape = MeanAbsolutePercentageError()
# mase = MeanAbsoluteScaledError()
# from sktime.forecasting.all import EnsembleForecaster
from sklearn.svm import SVR
# from sktime.transformations.series.detrend import ConditionalDeseasonalizer
# from sktime.datasets import load_macroeconomic


#%%
###### configurations for image quality#######
plt.rcParams["figure.figsize"] = [20, 8]   ##
# plt.rcParams['figure.dpi'] = 300           ## 300 for printing
plt.rc('font', size=8)                     ##
plt.rc('axes', titlesize=16)               ##
plt.rc('axes', labelsize=14)               ##
plt.rc('xtick', labelsize=10)              ##
plt.rc('ytick', labelsize=10)              ##
plt.rc('legend', fontsize=10)              ##
plt.rc('figure', titlesize=12)             ##
#############################################

#%%
# for better undrestanding I will define the funcions beside the codes
# path = Path('../input/af-timeseries-example/')
# Symbol = pd.read_csv(path.joinpath('Binance_SymbolUSDT_d.csv'),
#                   index_col='date',
#                   parse_dates=True,
#                   usecols=['close','date'])
# Symbol.columns = ['y']
# Symbol
today = date.today()
print("Today's date:", today)
TwoLastCandle = today - timedelta(days=3)
LastCandle = today - timedelta(days=7)

print ("TwoLastCandle : ",TwoLastCandle)
print ("LastCandle : ",LastCandle)

#%%
# EURUSD=X
# GPBUSD=X

# Symbol = pdr.get_data_yahoo([symb], start=datetime.datetime(2017, 1, 1))['Close']
df = myMT5Pro.getsymboldata("GBPUSD", "TIMEFRAME_D1", [2021,1,1,0,0,0], [2023, 4, 11,0,0,0], index_time=True,col_capitalize=True)

Symbol = df['Close']
Symbol_LastCandle = df[LastCandle-datetime.timedelta(days=1):LastCandle]['Close']


#%%
best_model = pd.DataFrame({'Model':[], 'Prediction':[]})

best_model.loc[len(best_model.index)] = ["***Real Today Price***", float(Symbol_LastCandle.iloc[-1])]
best_model.loc[len(best_model.index)] = ["***Real Last_day Price***", float(Symbol_LastCandle.iloc[-2])]

Symbol.columns = ['y']
Symbol
Symbol.plot(title='Symbol close price')
plt.show()

#%%
def handle_missing_data(df):
    n = int(df.isna().sum())
    if n > 0:
        print(f'found {n} missing observations...')
        df.ffill(inplace=True)
    else:
        print('no missing data')

Symbol_copy = Symbol.copy()
handle_missing_data(Symbol_copy)

Symbol_copy.isna().sum()

#%%
# 价格滞后作为特征
def one_step_forecast(df, window):
    d = df.values
    x = []
    n = len(df)
    idx = df.index[:-window]
    # 滞后5期
    for start in range(n-window):
        end = start + window
        x.append(d[start:end])
    cols = [f'x_{i}' for i in range(1, window+1)]
    x = np.array(x).reshape(n-window, -1)
    y = df.iloc[window:].values
    df_xs = pd.DataFrame(x, columns=cols, index=idx)
    df_y = pd.DataFrame(y.reshape(-1), columns=['y'], index=idx)
    return pd.concat([df_xs, df_y], axis=1).dropna()

Symbol_os = one_step_forecast(df=Symbol_copy, window=5)
print(Symbol_os.shape)

Symbol_copy.tail(10)
Symbol_os.tail(10)
Symbol_copy.tail(5)

#%%
def insert(df, row):
    insert_loc = df.index.max()

    if pd.isna(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row

def split_data(df, test_split=0.15):
    n = int(len(df) * test_split)
    train, test = df[:-n], df[-n:]
    return train, test

train, test = split_data(Symbol_os)
print(f'Train: {len(train)}, Test: {len(test)}')

#%%
class Standardize:
    def __init__(self, split=0.15):
        self.split = split

    def _transform(self, df):
        return (df - self.mu) / self.sigma

    def split_data(self, df, test_split=0.15):
        n = int(len(df) * test_split)
        train, test = df[:-n], df[-n:]
        return train, test

    def fit_transform(self, train, test):
        self.mu = train.mean()
        self.sigma = train.std()
        train_s = self._transform(train)
        test_s = self._transform(test)
        return train_s, test_s

    def transform(self, df):
        return self._transform(df)

    def inverse(self, df):
        return (df * self.sigma) + self.mu

    def inverse_y(self, df):
        return (df * self.sigma[0]) + self.mu[0]

#%%
scaler = Standardize()
train_s, test_s = scaler.fit_transform(train, test)
train_s.head()

y_train_original = scaler.inverse_y(train_s['y'])
train_original = scaler.inverse(train_s)
train_original.head()

df = Symbol.copy()

#%% One-Step Forecasting using Linear Regression Models with Scikit-Learn
Symbol_copy = Symbol.copy()
handle_missing_data(Symbol_copy)
# 10期滞后
Symbol_reg = one_step_forecast(Symbol_copy, 10)

# 把过去的特征放到当前，y设为0
df_tomorrow = pd.DataFrame(data=None, columns=Symbol_reg.columns)
insert(df_tomorrow, [float(Symbol_reg.iloc[-1].loc['x_2']),
                     float(Symbol_reg.iloc[-1].loc['x_3']),
                     float(Symbol_reg.iloc[-1].loc['x_4']),
                     float(Symbol_reg.iloc[-1].loc['x_5']),
                     float(Symbol_reg.iloc[-1].loc['x_6']),
                     float(Symbol_reg.iloc[-1].loc['x_7']),
                     float(Symbol_reg.iloc[-1].loc['x_8']),
                     float(Symbol_reg.iloc[-1].loc['x_9']),
                     float(Symbol_reg.iloc[-1].loc['x_10']),
                     float(Symbol_reg.iloc[-1].loc['y']),0])
df_tomorrow["date"] = Symbol_reg.index[-1]+ timedelta(days=1)
df_tomorrow.set_index('date', inplace = True)

df_tomorrow

print(Symbol_reg.shape)


#%%
from sklearn import metrics

# print ("Mean Absolute Error: ", metrics.mean_absolute_error(y_test , y_pred))
# print ("Mean Squared  Error: ", metrics.mean_squared_error(y_test , y_pred))
# print ("Root Absolute Error: ", np.sqrt(metrics.mean_squared_error(y_test , y_pred)))
# print ("R2 Score: ", metrics.r2_score(y_test , y_pred))

train_Symbol, test_Symbol = split_data(Symbol_reg, test_split=0.10)
scaler_Symbol = Standardize()
train_Symbol_s, test_Symbol_s = scaler_Symbol.fit_transform(train_Symbol,test_Symbol)

regressors = {
    'Linear Regression': LinearRegression(fit_intercept=False),
    'Elastic Net': ElasticNet(1, fit_intercept=False),
    'Ridge Regression': Ridge(1, fit_intercept=False),
    'Lasso Regression': Lasso(1, fit_intercept=False),
    'Huber Regression': HuberRegressor(fit_intercept=False)}

#%%
x_tomorrow = df_tomorrow.drop(columns=['y'])


def train_model(train, test, regressor, reg_name):
    X_train, y_train = train.drop(columns=['y']), train['y']
    X_test, y_test = test.drop(columns=['y']), test['y']

    print(f'training {reg_name} ...')

    regressor.fit(X_train, y_train)
    #     print (regressor)
    yhat = regressor.predict(X_test)
    #     print(X_test)
    #     print(X_test.shape)
    #     print(x_tomorrow.shape)
    y_tomorrow = regressor.predict(x_tomorrow)
    #     y_tomorrow = 0
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, yhat))
    mae_test = metrics.mean_absolute_error(y_test, yhat)
    mse_test = metrics.mean_squared_error(y_test, yhat)
    r2_test = metrics.r2_score(y_test, yhat)
    residuals = y_test.values - yhat

    model_metadata = {
        'Model Name': reg_name, 'Model': regressor,
        'RMSE': rmse_test, 'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test,
        'yhat': yhat, 'resid': residuals, 'actual': y_test.values, "y_tomorrow": float(y_tomorrow)}

    return model_metadata


def train_different_models(train, test, regressors):
    results = []
    for reg_name, regressor in regressors.items():
        results.append(train_model(train,
                                   test,
                                   regressor,
                                   reg_name))
    return results

Symbol_results = train_different_models(train=train_Symbol_s, test=test_Symbol_s, regressors=regressors)


#%% Evaluate
cols = ['Model Name', 'RMSE', 'MAE', 'MSE', 'R2']
Symbol_results = pd.DataFrame(Symbol_results)
Symbol_results[cols].sort_values('R2', ascending=False).style.background_gradient(cmap='summer_r')

Symbol.tail(5)
Symbol_LastCandle
print ("Real Price : ",float(Symbol_LastCandle.iloc[-1]))

cols = ['Model Name',"y_tomorrow"]
Symbol_results = pd.DataFrame(Symbol_results)
Symbol_results[cols].sort_values('y_tomorrow', ascending=False).style.background_gradient(cmap='summer_r')

for i in range(0,len(Symbol_results)):
    best_model.loc[len(best_model.index)] = [Symbol_results["Model Name"].iloc[i], Symbol_results['y_tomorrow'].iloc[i]]


#%% Plots
from statsmodels.graphics.tsaplots import plot_acf
def plot_results(cols, results, data_name):
    for row in results[cols].iterrows():
        yhat, resid, actual, name = row[1]
        plt.title(f'{data_name} - {name}')
        plt.plot(actual, 'k--', alpha=0.5)
        plt.plot(yhat, 'k')
        plt.legend(['actual', 'forecast'])
        plot_acf(resid, zero=False,
                 title=f'{data_name} - Autocorrelation')
        plt.show()
cols = ['yhat', 'resid', 'actual', 'Model Name']
plot_results(cols, Symbol_results, 'Symbol')


#%%
df_tomorrow = pd.DataFrame(data=None, columns=Symbol_reg.columns)
df_tomorrow["date"] = None
for i in range (1,4):
    insert(df_tomorrow, [float(Symbol_reg.iloc[-i].loc['x_2']), float(Symbol_reg.iloc[-i].loc['x_3']), float(Symbol_reg.iloc[-i].loc['x_4']), \
                         float(Symbol_reg.iloc[-i].loc['x_5']), float(Symbol_reg.iloc[-i].loc['x_6']), float(Symbol_reg.iloc[-i].loc['x_7']), \
                         float(Symbol_reg.iloc[-i].loc['x_8']), float(Symbol_reg.iloc[-i].loc['x_9']), float(Symbol_reg.iloc[-i].loc['x_10']),\
                         float(Symbol_reg.iloc[-i].loc['y']),float(Symbol_reg.iloc[-i].loc['y']) ,Symbol_reg.index[-i]+ timedelta(days=1)])

#     df_tomorrow["date"] = Symbol_reg.index[-1]+ timedelta(days=1)
df_tomorrow.set_index('date', inplace = True)

df_tomorrow


#%% Importing Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
# plt.rc("figure", figsize=(16, 4))

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,  SimpleRNN, Dropout, Input


import matplotlib
import tensorflow as tf

print(f'''
Matplotlib -> {matplotlib.__version__}
pandas -> {pd.__version__}   
numpy -> {np.__version__}
tensorflow -> {tf.__version__}
''')

#%%
Symbol_cp = Symbol.copy()


def handle_missing_data(df):
    n = int(df.isna().sum())
    if n > 0:
        print(f'found {n} missing observations...')
        df.ffill(inplace=True)


def one_step_forecast(df, window):
    d = df.values
    x = []
    n = len(df)
    idx = df.index[:-window]
    for start in range(n - window):
        end = start + window
        x.append(d[start:end])
    cols = [f'x_{i}' for i in range(1, window + 1)]
    x = np.array(x).reshape(n - window, -1)
    y = df.iloc[window:].values
    df_xs = pd.DataFrame(x, columns=cols, index=idx)
    df_y = pd.DataFrame(y.reshape(-1), columns=['y'], index=idx)
    return pd.concat([df_xs, df_y], axis=1).dropna()


def split_data(df, test_split=0.15):
    n = int(len(df) * test_split)
    train, test = df[:-n], df[-n:]
    return train, test


handle_missing_data(Symbol_cp)
Symbol_df = one_step_forecast(Symbol_cp, 10)

print(Symbol_df.shape)

#%%
class Standardize:
    def __init__(self, df, split=0.10):
        self.data = df
        self.split = split

    def split_data(self):
        n = int(len(self.data) * self.split)
        train, test = self.data.iloc[:-n], self.data.iloc[-n:]
        n = int(len(train) * self.split)
        train, val = train.iloc[:-n], train.iloc[-n:]
        assert len(test) + len(train) + len(val) == len(self.data)
        return train, test, val

    def _transform(self, data):
        data_s = (data - self.mu) / self.sigma
        return data_s

    def fit_transform(self):
        train, test, val = self.split_data()
        self.mu, self.sigma = train.mean(), train.std()
        train_s = self._transform(train)
        test_s = self._transform(test)
        val_s = self._transform(val)
        return train_s, test_s, val_s

    def inverse(self, data):
        return (data * self.sigma) + self.mu

    def inverse_y(self, data):
        return (data * self.sigma[-1]) + self.mu[-1]


scale_Symbol = Standardize(Symbol_df)

train_Symbol, test_Symbol, val_Symbol = scale_Symbol.fit_transform()

#%%
scale_df_tomorrow = Standardize(df_tomorrow, split=1)
train_Symbol_tomorrow, test_Symbol_tomorrow, val_Symbol_tomorrow = scale_df_tomorrow.fit_transform()
print(f'''
Symbol: train: {len(train_Symbol_tomorrow)} , test: {len(test_Symbol_tomorrow)}, val:{len(val_Symbol_tomorrow)}

''')
test_Symbol_tomorrow
df_tomorrow
df_tomorrow.mean()
df_tomorrow.std()

df_tomorrow_trns = (df_tomorrow - df_tomorrow.mean())/df_tomorrow.std()
df_tomorrow_trns.iloc[1]

train_Symbol_pt, test_Symbol_pt, val_Symbol_pt = scale_Symbol.fit_transform()
print(f'''
Symbol: train: {len(train_Symbol)} , test: {len(test_Symbol)}, val:{len(val_Symbol)}

''')

train_Symbol.head()
scale_Symbol.inverse(train_Symbol).head()

#%% Forecasting with Keras and PyTorch
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError, Accuracy, mape
from tensorflow.keras.layers import (BatchNormalization, Dense,
                TimeDistributed, Bidirectional,
                SimpleRNN, GRU, LSTM, Dropout)

#%% LSTM
def create_model(train, units, dropout=0.2):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=units,
                                input_shape=(train.shape[1],
                                             train.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    return model






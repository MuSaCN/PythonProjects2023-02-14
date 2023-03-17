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
import numpy as np


# %%

train = pd.read_csv(r'E:\学习\database数据集\金融中的机器学习\web-traffic-time-series-forecasting\train_1.csv').fillna(0)
train.head()


# %%

def parse_page(page):
    x = page.split('_')
    return ' '.join(x[:-3]), x[-3], x[-2], x[-1]


# %%

l = list(train.Page.apply(parse_page))
df = pd.DataFrame(l)
del l
df.columns = ['Subject', 'Sub_Page', 'Access', 'Agent']
df.head()

# %%

train = pd.concat([train, df], axis=1)
del train['Page']
del df

# %%

import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# %%

def lag_arr(arr, lag, fill):
    filler = np.full((arr.shape[0], lag, 1), -1)
    comb = np.concatenate((filler, arr), axis=1)
    result = comb[:, :arr.shape[1]]
    return result

# %%

def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0

# %%

def batc_autocorr(data, lag, series_length):
    corrs = []
    for i in range(data.shape[0]):
        c = single_autocorr(data, lag)
        corrs.append(c)
    corr = np.array(corrs)
    corr = corr.reshape(-1, 1)
    corr = np.expand_dims(corr, -1)
    corr = np.repeat(corr, series_length, axis=1)
    return corr


# %%
# 时间转字符串星期几
datetime.datetime.strptime(train.columns.values[0], '%Y-%m-%d').strftime('%a')
weekdays = [datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%a')
            for date in train.columns.values[:-4]]

day_one_hot = LabelEncoder().fit_transform(weekdays)
day_one_hot.shape
day_one_hot = day_one_hot.reshape(-1, 1)
day_one_hot.shape
day_one_hot = OneHotEncoder(sparse=False).fit_transform(day_one_hot)
day_one_hot.shape
day_one_hot = np.expand_dims(day_one_hot, 0)
day_one_hot.shape


# %%

agent_int = LabelEncoder().fit(train['Agent'])
agent_enc = agent_int.transform(train['Agent'])
agent_enc = agent_enc.reshape(-1, 1)
agent_one_hot = OneHotEncoder(sparse=False).fit(agent_enc)

del agent_enc

# %%

page_int = LabelEncoder().fit(train['Sub_Page'])
page_enc = page_int.transform(train['Sub_Page'])
page_enc = page_enc.reshape(-1, 1)
page_one_hot = OneHotEncoder(sparse=False).fit(page_enc)

del page_enc

# %%

acc_int = LabelEncoder().fit(train['Access'])
acc_enc = acc_int.transform(train['Access'])
acc_enc = acc_enc.reshape(-1, 1)
acc_one_hot = OneHotEncoder(sparse=False).fit(acc_enc)

del acc_enc


# %%


# %%

def get_batch(train, start=0, lookback=100):
    assert ((start + lookback) <= (train.shape[1] - 5)), 'End of lookback would be out of bounds'

    data = train.iloc[:, start:start + lookback].values
    target = train.iloc[:, start + lookback].values
    target = np.log1p(target)

    log_view = np.log1p(data)
    log_view = np.expand_dims(log_view, axis=-1)

    days = day_one_hot[:, start:start + lookback]
    days = np.repeat(days, repeats=train.shape[0], axis=0)

    year_lag = lag_arr(log_view, 365, -1)
    halfyear_lag = lag_arr(log_view, 182, -1)
    quarter_lag = lag_arr(log_view, 91, -1)

    agent_enc = agent_int.transform(train['Agent'])
    agent_enc = agent_enc.reshape(-1, 1)
    agent_enc = agent_one_hot.transform(agent_enc)
    agent_enc = np.expand_dims(agent_enc, 1)
    agent_enc = np.repeat(agent_enc, lookback, axis=1)

    page_enc = page_int.transform(train['Sub_Page'])
    page_enc = page_enc.reshape(-1, 1)
    page_enc = page_one_hot.transform(page_enc)
    page_enc = np.expand_dims(page_enc, 1)
    page_enc = np.repeat(page_enc, lookback, axis=1)

    acc_enc = acc_int.transform(train['Access'])
    acc_enc = acc_enc.reshape(-1, 1)
    acc_enc = acc_one_hot.transform(acc_enc)
    acc_enc = np.expand_dims(acc_enc, 1)
    acc_enc = np.repeat(acc_enc, lookback, axis=1)

    year_autocorr = batc_autocorr(data, lag=365, series_length=lookback)
    halfyr_autocorr = batc_autocorr(data, lag=182, series_length=lookback)
    quarter_autocorr = batc_autocorr(data, lag=91, series_length=lookback)

    medians = np.median(data, axis=1)
    medians = np.expand_dims(medians, -1)
    medians = np.expand_dims(medians, -1)
    medians = np.repeat(medians, lookback, axis=1)

    '''
    print(log_view.shape)
    print(days.shape)
    print(year_lag.shape)
    print(halfyear_lag.shape)
    print(page_enc.shape)
    print(agent_enc.shape)
    print(acc_enc.shape)'''

    batch = np.concatenate((log_view,
                            days,
                            year_lag,
                            halfyear_lag,
                            quarter_lag,
                            page_enc,
                            agent_enc,
                            acc_enc,
                            year_autocorr,
                            halfyr_autocorr,
                            quarter_autocorr,
                            medians), axis=2)

    return batch, target


# %%

def generate_batches(train, batch_size=32, lookback=100):
    num_samples = train.shape[0]
    num_steps = train.shape[1] - 5
    while True:
        for i in range(num_samples // batch_size): # i=0
            batch_start = i * batch_size
            batch_end = batch_start + batch_size

            seq_start = np.random.randint(num_steps - lookback)
            X, y = get_batch(train=train.iloc[batch_start:batch_end], start=seq_start)
            yield X, y


# %%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Activation, GlobalMaxPool1D, Flatten

# %%

max_len = 100
n_features = 29

# %%

model = Sequential()

model.add(Conv1D(16, 5, input_shape=(max_len, n_features)))
model.add(Activation('relu'))
model.add(MaxPool1D(5))

model.add(Conv1D(16, 5))
model.add(Activation('relu'))
model.add(MaxPool1D(5))

model.add(Flatten())
model.add(Dense(1))

# %%

model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

# %%

from sklearn.model_selection import train_test_split

# %%

batch_size = 128
train_df, val_df = train_test_split(train, test_size=0.1)
train_gen = generate_batches(train=train_df, batch_size=batch_size)
val_gen = generate_batches(train=val_df, batch_size=batch_size)

n_train_samples = train_df.shape[0]
n_val_samples = val_df.shape[0]

# %%


# %%

a, b = next(train_gen)

# %%

model.fit_generator(train_gen,
                    epochs=1,
                    steps_per_epoch=n_train_samples // batch_size,
                    validation_data=val_gen,
                    validation_steps=n_val_samples // batch_size)

# %%

from tensorflow.keras.layers import SimpleRNN

model = Sequential()
model.add(SimpleRNN(16, input_shape=(max_len, n_features)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

# %%

model.fit_generator(train_gen,
                    epochs=1,
                    steps_per_epoch=n_train_samples // batch_size,
                    validation_data=val_gen,
                    validation_steps=n_val_samples // batch_size)

# %%

from keras.layers import SimpleRNN

model = Sequential()
model.add(SimpleRNN(32, return_sequences=True, input_shape=(max_len, n_features)))
model.add(SimpleRNN(16, return_sequences=True))
model.add(SimpleRNN(16))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

# %%

model.fit_generator(train_gen,
                    epochs=1,
                    steps_per_epoch=n_train_samples // batch_size,
                    validation_data=val_gen,
                    validation_steps=n_val_samples // batch_size)

# %%

from keras.layers import CuDNNLSTM

model = Sequential()
model.add(CuDNNLSTM(16, input_shape=(max_len, n_features)))
model.add(Dense(1))

# %%

model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

# %%

model.fit_generator(train_gen,
                    epochs=1,
                    steps_per_epoch=n_train_samples // batch_size,
                    validation_data=val_gen,
                    validation_steps=n_val_samples // batch_size)

# %%

from keras.layers import LSTM

model = Sequential()
model.add(LSTM(16,
               recurrent_dropout=0.1,
               return_sequences=True,
               input_shape=(max_len, n_features)))

model.add(LSTM(16, recurrent_dropout=0.1))

model.add(Dense(1))

# %%

model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

# %%

model.fit_generator(train_gen,
                    epochs=1,
                    steps_per_epoch=n_train_samples // batch_size,
                    validation_data=val_gen,
                    validation_steps=n_val_samples // batch_size)

# %%



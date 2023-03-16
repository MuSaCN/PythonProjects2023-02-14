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
myDefault.set_backend_default("tkagg")

# %%

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


# %%

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# %%

train = pd.read_csv(r'E:\学习\database数据集\金融中的机器学习\web-traffic-time-series-forecasting\train_1.csv').fillna(0)
train.head()


# %%

def parse_page(page):
    x = page.split('_')
    return ' '.join(x[:-3]), x[-3], x[-2], x[-1]


# %%

parse_page(train.Page[0])

# %%

l = list(train.Page.apply(parse_page))
df = pd.DataFrame(l)
df.columns = ['Subject', 'Sub_Page', 'Access', 'Agent']
df.head()

# %%

train = pd.concat([train, df], axis=1)
del train['Page']

# %%

fig, ax = plt.subplots(figsize=(10, 7))
train.Sub_Page.value_counts().plot(kind='bar')
plt.show()

# %%

fig, ax = plt.subplots(figsize=(10, 7))
train.Access.value_counts().plot(kind='bar')
plt.show()

# %%

fig, ax = plt.subplots(figsize=(10, 7))
train.Agent.value_counts().plot(kind='bar')
plt.show()

# %%

train.head()

# %%

from matplotlib import dates

idx = 39457
window = 10

data = train.iloc[idx, 0:-4]
name = train.iloc[idx, -4]
days = [r for r in range(data.shape[0])]

fig, ax = plt.subplots(figsize=(10, 7))

plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title(name)


ax.plot(days, data.values, color='grey')
# 这里是用卷积的方式求移动平均，麻烦了。
# ax.plot(np.convolve(data, np.ones((window,)) / window, mode='valid'), color='black')
moveavg = data.rolling(window=10).mean()
# 注意，这里没有把移动平均做时间对齐。
# ax.plot(moveavg.values, color='black')
ax.plot(moveavg.dropna().values, color='black')


ax.set_yscale('log')
plt.show()

# %%

fig, ax = plt.subplots(figsize=(10, 7))
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Twenty One Pilots Popularity')
ax.set_yscale('log')
handles = []
for country in ['de', 'en', 'es', 'fr', 'ru']: # country="en"
    idx = np.where((train['Subject'] == 'Twenty One Pilots')
                   & (train['Sub_Page'] == '{}.wikipedia.org'.format(country))
                   & (train['Access'] == 'all-access') & (train['Agent'] == 'all-agents'))
    idx = idx[0][0]

    data = train.iloc[idx, 0:-4]
    handle = ax.plot(days, data.values, label=country)
    handles.append(handle)

ax.legend()
plt.show()

# %%
# 引入快速傅里叶变换
from scipy.fftpack import fft, ifft

x = np.arange(5)
x = ifft(x)
fft(x)


# %%

# idx = 39457
data = train.iloc[:, 0:-4]
data.shape
fft_complex = fft(data) # 默认以最后的维度做傅里叶变换，此处为以行数据做

# %%

fft_complex.shape

# %%
# 转成复数的模
fft_mag = [np.sqrt(np.real(x) * np.real(x) +
                   np.imag(x) * np.imag(x)) for x in fft_complex]

# %%

arr = np.array(fft_mag)
arr.shape

# %%
# 以列求均值.
fft_mean = np.mean(arr, axis=0)
fft_mean.shape

# %%
# 生成0-1之间550个数据
fft_xvals = [day / fft_mean.shape[0] for day in range(fft_mean.shape[0])]

# %%

npts = len(fft_xvals) // 2 + 1
fft_mean = fft_mean[:npts]
fft_xvals = fft_xvals[:npts]

# %%

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(fft_xvals[1:], fft_mean[1:])
plt.axvline(x=1. / 7, color='red', alpha=0.3)
plt.axvline(x=2. / 7, color='red', alpha=0.3)
plt.axvline(x=3. / 7, color='red', alpha=0.3)
plt.show()

# %%

from pandas.plotting import autocorrelation_plot

# %%

plt.figure(figsize=(10, 7))
a = np.random.choice(data.shape[0], 1000)

for i in a:
    autocorrelation_plot(data.iloc[i])
autocorrelation_plot(data.iloc[0])
plt.title('1K Autocorrelations')
plt.show()

# %%

fig = plt.figure(figsize=(10, 7))

autocorrelation_plot(data.iloc[110])
plt.title(' '.join(train.loc[110, ['Subject', 'Sub_Page']]))
plt.show()

# %%

data.shape

# %%

from sklearn.model_selection import train_test_split

# %%

X = data.iloc[:, :500]
y = data.iloc[:, 500:]

# %%

X.shape

# %%

y.shape

# %%

X_train, X_val, y_train, y_val = train_test_split(X.values, y.values,
                                                  test_size=0.1,
                                                  random_state=42)


# %%
# 均匀绝对百分比误差，eps用于防止分母为0
def mape(y_true, y_pred):
    eps = 1
    err = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    return err


# %%
# 仅用中位数进行预测
lookback = 50
lb_data = X_train[:, -lookback:]
med = np.median(lb_data, axis=1, keepdims=True)
err = mape(y_train, med)

# %%

err

# %%

idx = 15000

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(np.arange(500), X_train[idx], label='X')
ax.plot(np.arange(500, 550), y_train[idx], label='True')

ax.plot(np.arange(500, 550), np.repeat(med[idx], 50), label='Forecast')

plt.title(' '.join(train.loc[idx, ['Subject', 'Sub_Page']]))
ax.legend()
ax.set_yscale('log')
plt.show()

# %%

from statsmodels.tsa.arima.model import ARIMA

# %%

model = ARIMA(X_train[0], order=(5, 1, 5))

# %%

model = model.fit()

# %%

model.summary()

# %%

fig, ax = plt.subplots(figsize=(10, 7))
idx = 0
residuals = pd.DataFrame(model.resid)
ax.plot(residuals)

plt.title('ARIMA residuals for 2NE1 pageviews')
plt.show()

# %%

residuals.plot(kind='kde',
               figsize=(10, 7),
               title='ARIMA residual distribution 2NE1 ARIMA', legend=False)
plt.show()

# %%

# predictions, stderr, conf_int = model.forecast(50)
predictions = model.forecast(50)

# %%

# target = y_train[0]
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(np.arange(480, 500), X_train[0, 480:], label='X')
ax.plot(np.arange(500, 550), y_train[0], label='True')

ax.plot(np.arange(500, 550), predictions, label='Forecast')

plt.title('2NE1 ARIMA forecasts')
ax.legend()
ax.set_yscale('log')
plt.show()

# %%

import simdkalman

# %%

smoothing_factor = 5.0

n_seasons = 7

# --- define state transition matrix A
state_transition = np.zeros((n_seasons + 1, n_seasons + 1))
# hidden level
state_transition[0, 0] = 1
# season cycle
state_transition[1, 1:-1] = [-1.0] * (n_seasons - 1)
state_transition[2:, 1:-1] = np.eye(n_seasons - 1)

# %%

state_transition

# %%

observation_model = [[1, 1] + [0] * (n_seasons - 1)]

# %%

observation_model

# %%

level_noise = 0.2 / smoothing_factor
observation_noise = 0.2
season_noise = 1e-3

process_noise_cov = np.diag([level_noise, season_noise] + [0] * (n_seasons - 1)) ** 2
observation_noise_cov = observation_noise ** 2

# %%

process_noise_cov

# %%

observation_noise_cov

# %%

kf = simdkalman.KalmanFilter(state_transition=state_transition,
                             process_noise=process_noise_cov,
                             observation_model=observation_model,
                             observation_noise=observation_noise_cov)

# %%

result = kf.compute(X_train[0], 50)

# %%

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(np.arange(480, 500), X_train[0, 480:], label='X')
ax.plot(np.arange(500, 550), y_train[0], label='True')

ax.plot(np.arange(500, 550),
        result.predicted.observations.mean,
        label='Predicted observations')

ax.plot(np.arange(500, 550),
        result.predicted.states.mean[:, 0],
        label='redicted states')

ax.plot(np.arange(480, 500),
        result.smoothed.observations.mean[480:],
        label='Expected Observations')

ax.plot(np.arange(480, 500),
        result.smoothed.states.mean[480:, 0],
        label='States')

ax.legend()
ax.set_yscale('log')
plt.show()

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
# 外汇交易策略和RSI：
# 为了识别外汇交易趋势，我们不应该仅仅依靠价格的上升/下降趋势，还应该考虑在一个范围内（如5个之前的价格和5个之后的价格）的局部最大值和最小值来判断相对强弱指数（RSI）的上升/下降趋势。如果价格和RSI都显示上升趋势，那么它就是一个真正的峰值。如果价格显示上升趋势，但RSI显示下降趋势，那么它是一个负的背离。RSI被认为是一种良好的外汇交易策略。

# Investopedia的定义： RSI是一个动量指标，它比较了特定时间段内最近的涨跌幅度，以衡量价格运动的速度和变化。它主要用于识别资产交易中的超买或超卖情况。

#%%
# importing data
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = myMT5Pro.getsymboldata("EURUSD", "TIMEFRAME_H4", [2003,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=False)
df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
df.columns=['time', 'open', 'high', 'low', 'close', 'volume']

df=df[df['volume']!=0] # discard volume zero data points

df.reset_index(drop=True, inplace=True)
df.isna().sum()
df.tail() # printing tail just to check how many rows are present

#%%
# 技术分析和指标
# "pandas_ta "模块有一个内置的RSI指标，它使用指数移动平均线而不是绝对价格。因此，它是平滑的RSI。在这里，我们也将定义我们的自定义RSI，它使用蜡烛的收盘值。这将给出RSI的更多极端值。我们的目的是检查使用两个版本的RSI的结果：平滑的和极端的。我们可以使用在14到20之间徘徊的任何窗口大小。
import talib


# df.ta.rsi is predefined in TA module. It is "smoothened RSI". It uses exponential moving average instead of the absolute price
# 14 is the default window
df['RSI'] = talib.RSI(df["close"], timeperiod = 14)
df.head(15) # we will get RSI value after the 14th row since we fixed our window as 14


#%%
# Redefined RSI: our custom RSI which uses the closing value of the candles. This will give more extreme values of the RSI
# we are taking 20 as the window
def customRSI(price, n=20):
    delta = price['close'].diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(window=n).mean()
    RolDown = dDown.rolling(window=n).mean().abs()

    RS = RolUp / RolDown
    rsi = 100.0 - (100.0 / (1.0 + RS))
    return rsi


df['custom_RSI'] = customRSI(df)
# df.dropna(inplace=True)
# df.reset_index(drop=True, inplace=True)

df.head(21) # we will get RSI value after the 20th row since we fixed our window as 20


#%%
# Visualize Smoothened RSI for a Long Horizon
# We will take a slice of last 825 data points to see how smoothened RSI (pandas_ta predefined RSI) looks like
# taking a slice from dataset

dfpl = df[28000:28825]

import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

fig = make_subplots(rows=2, cols=1)
fig.append_trace(go.Candlestick(x=dfpl.index,
                open=dfpl['open'],
                high=dfpl['high'],
                low=dfpl['low'],
                close=dfpl['close']), row=1, col=1)
fig.append_trace(go.Scatter(
    x=dfpl.index,
    y=dfpl['RSI'],
), row=2, col=1)

fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()

#%%
# Visualize Extreme RSI for a Long Horizon
# We will take a slice of last 825 data points to see how extreme RSI (our custom version of RSI) looks like. Not much difference is visible, right?

# taking a slice from dataset

dfpl = df[28000:28825]
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

fig = make_subplots(rows=2, cols=1)
fig.append_trace(go.Candlestick(x=dfpl.index,
                open=dfpl['open'],
                high=dfpl['high'],
                low=dfpl['low'],
                close=dfpl['close']), row=1, col=1)
fig.append_trace(go.Scatter(
    x=dfpl.index,
    y=dfpl['custom_RSI'],
), row=2, col=1)

fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()

















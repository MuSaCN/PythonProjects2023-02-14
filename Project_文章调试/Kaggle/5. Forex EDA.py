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
# 简介
# 外汇是一个大型的全球市场，允许人们进行货币之间的交易。作为世界上最大的市场，它拥有单日近7万亿美元的交易量。随着人工智能和机器学习的普及，许多人试图预测未来的货币价格，然而，许多人几乎没有成功。
# 预测金融市场类似于预测未来。由于有这么多未知和不可预测的因素，建立一个机器学习模型来预测未来事件的发生实在是太不可能了（就目前而言）。因此，本笔记本没有试图预测未来的价格，而是对货币市场进行了简单的分析（只分析了一些货币对），它与更广泛的市场的相关性，也许还有我们最近观察到的一些趋势。

# 在本节中，我们将对我们的数据做一个简单的分析，这可能有助于我们以后的数据探索。

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import os
from datetime import datetime

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px

from plotly.subplots import make_subplots
from plotly.graph_objs import Line

from scipy import stats
import seaborn as sns

# Data Paths
# Data Paths
daily_eurusd_df = pd.read_csv("../input/xauusdxaueureurusd-daily/data/EUR_USD Historical Data.csv")
xau_eur_df = pd.read_csv("../input/xauusdxaueureurusd-daily/data/XAU_EUR Historical Data (2).csv")
xau_usd_df = pd.read_csv("../input/xauusdxaueureurusd-daily/data/XAU_USD Historical Data (1).csv")
oil_df = pd.read_csv("../input/crude-oil-prices/Oil_Prices.csv")
usd_index_df = pd.read_csv("../input/us-dollar-index/US Dollar Index Futures Historical Data.csv")
us_interest_rates = pd.read_csv("../input/historical-fed-funds/fed-funds-rate-historical-chart_Mar2021.csv")
gold_prices_df = pd.read_csv("../input/gold-and-silver-prices-dataset/gold_price.csv")
daily_usdjpy_df = pd.read_csv("../input/usdjpy-historical-data-2014-2021/USD_JPY Historical Data.csv")
vix_df = pd.read_csv("../input/cboe-vix-historical-data-2014-2021/CBOE Volatility Index Historical Data.csv")
snp_500_df = pd.read_csv("../input/sp-500-historical-data-2014-2021/SP 500 Historical Data.csv")



























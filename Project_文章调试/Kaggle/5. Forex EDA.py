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
mypltly = MyPlot.MyClass_Plotly() # plotly画图相关
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
import warnings
warnings.filterwarnings('ignore')
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
daily_eurusd_df = myMT5Pro.getsymboldata("EURUSD", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
xau_eur_df = myMT5Pro.getsymboldata("XAUEUR", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
xau_usd_df = myMT5Pro.getsymboldata("XAUUSD", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
oil_df = myMT5Pro.getsymboldata("XTIUSD", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
usd_index_df = myMT5Pro.getsymboldata("USDX.index", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
us_interest_rates = None
gold_prices_df = myMT5Pro.getsymboldata("XAUUSD", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
daily_usdjpy_df = myMT5Pro.getsymboldata("USDJPY", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
vix_df = None
snp_500_df = myMT5Pro.getsymboldata("US500", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)

#%%
# Renaming Columns For Merging Later on
daily_eurusd_df.rename(columns = {'Close' : 'EURUSD_Price', 'Open' : 'EURUSD_Open', "High":"EURUSD_High", "Low":"EURUSD_Low", "Rate":"EURUSD_Change%"}, inplace = True)
xau_usd_df.rename(columns = {'Close' : 'XAUUSD_Price', 'Open' : 'XAUUSD_Open', "High":"XAUUSD_High", "Low":"XAUUSD_Low", "Rate":"XAUUSD_Change%"}, inplace = True)
xau_eur_df.rename(columns = {'Close' : 'XAUEUR_Price', 'Open' : 'XAUEUR_Open', "High":"XAUEUR_High", "Low":"XAUEUR_Low", "Rate":"XAUEUR_Change%"}, inplace = True)


def modify_datetime(df_column):
    """
    Changes Date Format from Feb 08, 2020 --> 08/02/2020 [dd/mm/YYYY]
    """
    df_column["Time"] = df_column["Time"].apply(lambda x: x.strftime("%d/%m/%Y"))
    # df_column["Time"] = df_column["Time"].apply(lambda x:datetime.strptime(x.lower().replace(",", ""), "%b %d %Y").strftime("%d/%m/%Y"))
    return df_column["Time"]


def remove_comma(df_column, column_name):
    """
    Removes Comma from Prices E.g [1,234,234 --> 1234234]
    """
    try:
        df_column[column_name] = df_column[column_name].apply(lambda x: x.replace(",", ""))
        return df_column[column_name]
    except:
        return df_column[column_name]


daily_eurusd_df["Date"] = modify_datetime(df_column=daily_eurusd_df)
print("No. of Data Points (EURUSD) :", len(daily_eurusd_df))

xau_usd_df["Date"] = modify_datetime(xau_usd_df)
print("No. of Data Points (XAUUSD) :", len(xau_usd_df))

xau_eur_df["Date"] = modify_datetime(xau_eur_df)
print("No. of Data Points (XAUEUR) :", len(xau_eur_df))


# Merging all the Dataframes together
merge_df = pd.merge(daily_eurusd_df, xau_usd_df, how="outer", on="Date")
merge_df = pd.merge(merge_df, xau_eur_df, how="outer", on="Date")
# Re-Fromatting Dataframe
merge_df.dropna(inplace=True)
merge_df = merge_df[::-1].reset_index()
del merge_df["index"]
# Removes Commas from Columns we need
merge_df["XAUUSD_Price"] = remove_comma(merge_df, "XAUUSD_Price")
merge_df["XAUEUR_Price"] = remove_comma(merge_df, "XAUEUR_Price")
# Make an archive/copyy of the original dataframe
_merge_df = merge_df.copy()


#%% Statistics
"""Mean Price"""

mean_eurusd = merge_df["EURUSD_Price"].mean()
mean_xauusd = merge_df["XAUUSD_Price"].astype(np.float).mean()
mean_xaueur = merge_df["XAUEUR_Price"].astype(np.float).mean()


"""众数 Mode Price"""

mode_eurusd = merge_df["EURUSD_Price"].mode().tolist()
mode_xauusd = merge_df["XAUUSD_Price"].mode().astype(float).tolist()
mode_xaueur = merge_df["XAUEUR_Price"].mode().astype(float).tolist()


"""Plotting Candlestick Graphs with Mean and Mode Values"""

fig = go.Figure(data=[go.Candlestick(x=merge_df['Date'],
                open=merge_df['EURUSD_Open'],
                high=merge_df['EURUSD_High'],
                low=merge_df['EURUSD_Low'],
                close=merge_df['EURUSD_Price'],
                name="Candlestick Graph")])

for i in mode_eurusd:
    x = np.array(["02/01/2014", "08/02/2021"])
    y = np.array([i, i])
    fig.add_trace(go.Scatter(x=x, y=y, name="Mode Value(s)",mode='lines'))

x = np.array(["02/01/2014", "08/02/2021"])
y = np.array([mean_eurusd, mean_eurusd])
fig.add_trace(go.Scatter(x=x, y=y, name="Mean Value",line=dict(color='red', width=1.5, dash='dot')))

fig.update_layout(showlegend=True)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(height=600, width=1000, title_text="EURUSD Chart")

fig.show()
print("Mean EURUSD Price (Jan 2014 - Feb 2021) :", round(mean_eurusd, 4))
print("Mode EURUSD Price(s) (Jan 2014 - Feb 2021) :", mode_eurusd)
mypltly.plot_on_webpage(fig)

#%%
# Overview of Data
sns.displot(merge_df['EURUSD_Price'])

"""
Skewness is a measure of the symmetrical nature of data. 
Kurtosis is a measure of how heavy-tailed or light-tailed the data is relative to a normal distribution.
"""
print("Skewness: %f" % merge_df['EURUSD_Price'].skew())
print("Kurtosis: %f" % merge_df['EURUSD_Price'].kurt())
plt.show()

#%% Data Exploration
# 货币强度与黄金的关系
# 货币强度经常被用作辅助交易的指标。然而，有许多方法来定义货币的强度。一些可用的开源货币强度表测量每种货币相对于美元的强度，然后对主要货币对进行相应排名。由于美元的影响很大，因此美国发生的事件的影响也更大，这对货币强度的看法更加偏颇。例如，澳元兑美元的上涨并不一定意味着澳元的改善，而可能是美国正在发生的负面的基本面变化。因此，我们必须避免使用其他货币作为衡量货币强度的标准。
# 也许从黄金的角度看货币，可以提供一个不太偏颇的货币强度前景。在本节中，我们看一下欧元/美元与黄金的货币强势有多密切相关。
fig = go.Figure(data=[go.Candlestick(x=merge_df['Date'],
                open=merge_df['XAUUSD_Open'],
                high=merge_df['XAUUSD_High'],
                low=merge_df['XAUUSD_Low'],
                close=merge_df['XAUUSD_Price'],
                name="Candlestick Graph")])

for i in mode_xauusd:
    x = np.array(["02/01/2014", "08/02/2021"])
    y = np.array([i, i])
    fig.add_trace(go.Scatter(x=x, y=y, name="Mode Value(s)",mode='lines'))

x = np.array(["02/01/2014", "08/02/2021"])
y = np.array([mean_xauusd, mean_xauusd])
fig.add_trace(go.Scatter(x=x, y=y, name="Mean Value",line=dict(color='red', width=1.5, dash='dot')))

fig.update_layout(showlegend=True)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(height=600, width=1000, title_text="XAUUSD Chart")

fig.show()
print("Mean XAUUSD Price (Jan 2014 - Feb 2021) :", round(mean_xauusd, 2))
print("Mode XAUUSD Price(s) (Jan 2014 - Feb 2021) :", mode_xauusd)
mypltly.plot_on_webpage(fig)


fig = go.Figure(data=[go.Candlestick(x=merge_df['Date'],
                open=merge_df['XAUEUR_Open'],
                high=merge_df['XAUEUR_High'],
                low=merge_df['XAUEUR_Low'],
                close=merge_df['XAUEUR_Price'],
                name="Candlestick Graph")])

for i in mode_xaueur:
    x = np.array(["02/01/2014", "08/02/2021"])
    y = np.array([i, i])
    fig.add_trace(go.Scatter(x=x, y=y, name="Mode Value(s)",mode='lines'))

x = np.array(["02/01/2014", "08/02/2021"])
y = np.array([mean_xaueur, mean_xaueur])
fig.add_trace(go.Scatter(x=x, y=y, name="Mean Value",line=dict(color='red', width=1.5, dash='dot')))

fig.update_layout(showlegend=True)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(height=600, width=1000, title_text="XAUEUR Chart")

fig.show()
print("Mean XAUEUR Price (Jan 2014 - Feb 2021) :", round(mean_xaueur, 2))
print("Mode XAUEUR Price(s) (Jan 2014 - Feb 2021) :", mode_xaueur)
mypltly.plot_on_webpage(fig)

#%%
# Finding the Difference Between XAUUSD and XAUEUR
merge_df["XAUUSD_XAUEUR_Diff_Price"] = (merge_df["XAUUSD_Price"].astype(float) - merge_df["XAUEUR_Price"].astype(float))
merge_df["XAUEUR / XAUUSD Price"] = (merge_df["XAUUSD_Price"].astype(float) / merge_df["XAUEUR_Price"].astype(float))

# EUR and USD Currency Strength is taken as (XAUEUR - XAUUSD) in this case
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 用 XAUUSD_XAUEUR_Diff_Price 表示 EUR和USD的货币强度.
# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.XAUUSD_XAUEUR_Diff_Price, name="EUR and USD Currency Strength"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.EURUSD_Price, name="EUR/USD"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="EUR/USD Versus EUR and USD Currency Strength"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>EUR and USD Currency Strength</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>EUR/USD</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)
print("Correlation Between Currency Strength of EUR and USD (XAUEUR - XAUUSD) and EUR/USD :", round(stats.pearsonr(merge_df.XAUUSD_XAUEUR_Diff_Price, merge_df.EURUSD_Price)[0],4))
# 我们可以看到欧元和美元的货币强度与欧元/美元价格密切相关。从2015年开始，它们的走势甚至似乎是相互同步的。

#%% 市场间关系[商品]
# 长期以来，黄金和石油等大宗商品一直是货币的衡量标准，无论是通过直接影响货币价格的手段，还是通过其与利率的相关性，似乎大宗商品在衡量外汇市场的未来走势中发挥了至关重要的作用。让我们看看黄金和石油与我们手头的一些数据的相关性.
# Oil Dataframe
oil_df = myMT5Pro.getsymboldata("XTIUSD", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
oil_df.rename(columns = {'Close' : 'Oil_Price', 'Tick_volume' : 'Oil_Volume', "Open": "Oil_Open", "High":"Oil_High", "Low":"Oil_Low"}, inplace=True)
oil_df = oil_df[::-1].reset_index()
# del oil_df["index"]
oil_df = oil_df[712:]
oil_df["Date"] = oil_df["Time"].apply(lambda x:x.strftime("%d/%m/%Y"))
oil_df.reset_index()

# Gold Dataframe
gold_prices_df = myMT5Pro.getsymboldata("XAUUSD", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
gold_prices_df.rename(columns={"Time":"Date", "Close": "Gold_Price"}, inplace=True)
gold_prices_df["Date"] = gold_prices_df["Date"].apply(lambda x:x.strftime("%d/%m/%Y"))
gold_prices_df = gold_prices_df.dropna()
# us_interest_rates = us_interest_rates[us_interest_rates['Date'].between("02/01/2014", "08/02/2021")]
start_date = "02/01/2014"
end_date = "08/02/2021"
gold_prices_df = gold_prices_df[gold_prices_df[gold_prices_df.Date==(start_date)].index[0] : gold_prices_df[gold_prices_df.Date==(end_date)].index[0]+pd.to_timedelta(1, unit='D')].reset_index()

# USDX Dataframe
usd_index_df = myMT5Pro.getsymboldata("USDX.index", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
usd_index_df.rename(columns = {"Time":"Date",'Close' : 'USDX_Price', "Open": "USDX_Open", "High":"USDX_High", "Low":"USDX_Low", "Tick_volume":"USDX_Vol", "Rate":"USDX_Change%"}, inplace=True)
usd_index_df["Date"] = usd_index_df["Date"].apply(lambda x:x.strftime("%d/%m/%Y"))

# US Interest Rates
us_interest_rates = pd.read_csv(__mypath__.get_current_workpath() + r"\Project_文章调试\Kaggle\data\fed-funds-rate-historical-chart_Mar2021.csv")
us_interest_rates.rename(columns={"date":"Date", " value": "US_Interest_Rates_Value"}, inplace=True)
us_interest_rates["Date"] = us_interest_rates["Date"].apply(lambda x:datetime.strptime(x, "%m/%d/%Y").strftime("%d/%m/%Y"))
us_interest_rates = us_interest_rates.dropna()
# us_interest_rates = us_interest_rates[us_interest_rates['Date'].between("02/01/2014", "08/02/2021")]
start_date = "02/01/2014"
end_date = "08/02/2021"
us_interest_rates = us_interest_rates[us_interest_rates[us_interest_rates.Date==(start_date)].index[0] : us_interest_rates[us_interest_rates.Date==(end_date)].index[0]+1].reset_index().drop("index", axis=1)

# Merging Dataframe
merge_df = pd.merge(merge_df, oil_df, how="left", on="Date")
merge_df = pd.merge(merge_df, gold_prices_df, how="left", on="Date")
merge_df = pd.merge(merge_df, usd_index_df, how="left", on="Date")
merge_df = pd.merge(merge_df, us_interest_rates, how="left", on="Date")
merge_df["Gold/Oil"] = merge_df["Gold_Price"] / merge_df["Oil_Price"]

#%%
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Gold_Price, name="Gold Price"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.EURUSD_Price, name="EURUSD Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Gold Prices Versus EURUSD Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Gold</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>EURUSD</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Gold_Price, name="Gold Price"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.XAUUSD_XAUEUR_Diff_Price, name="XAUEUR - XAUUSD Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Gold Prices Versus XAUEUR - XAUUSD Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Gold</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>XAUUSD_XAUEUR_Diff</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)



# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Gold_Price, name="Gold Price"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df["XAUEUR / XAUUSD Price"], name="XAUEUR / XAUUSD Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Gold Prices Versus XAUEUR/XAUUSD Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Gold</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>XAUEUR/XAUUSD</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)



# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Gold_Price, name="Gold Price"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.USDX_Price, name="USDX Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Gold Prices Versus USDX Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Gold</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>USDX</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)

#%% # 分析黄金
merge_df.corr(method='pearson')
corr_df = merge_df[["Gold_Price", "USDX_Price", "EURUSD_Price", "XAUEUR / XAUUSD Price", "XAUUSD_XAUEUR_Diff_Price"]]

# Correlation Heatmap
corrmat = corr_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
ax.set_title("Correlation Heatmap")
sns.heatmap(corrmat, square=True, annot=True)
plt.show()

#%%
xau_eur_df.XAUEUR_Price.corr(xau_usd_df.XAUUSD_Price)

# **反常现象**
# 有趣的是，黄金与欧元兑美元和 XAUUSD/XAUEUR 的相关性只有0.15左右，但与 XAUEUR-XAUUSD 的相关性为0.58。从图表中，我们可以清楚地看到在以下时期，XAUEUR - XAUUSD和（欧元兑美元和XAUUSD / XAUEUR）之间的差异：
# 2016年6月 - 2016年12月
# 2020年3月 - 2021年2月
# 一般来说，对XAUEUR - XAUUSD的影响比XAUEUR / XAUUSD或欧元/美元的影响要大，因为两个值的减法的影响通常比两个值的除法要大（例如黄金和欧元/美元）。考虑到在这两个时期发生的一些全球事件，很容易理解为什么我们的相关性在两个非常相似的价值之间看到如此大的差异。

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Oil_Price, name="Oil Price"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.EURUSD_Price, name="EURUSD Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Oil Prices Versus EURUSD Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Oil</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>EURUSD</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Oil_Price, name="Oil Price"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.XAUUSD_XAUEUR_Diff_Price, name="XAUEUR - XAUUSD Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Oil Prices Versus XAUEUR - XAUUSD Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Oil</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>XAUUSD_XAUEUR_Diff</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Oil_Price, name="Oil Price"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df["XAUEUR / XAUUSD Price"], name="XAUEUR / XAUUSD Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Oil Prices Versus XAUEUR/XAUUSD Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Oil</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>XAUEUR/XAUUSD</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Oil_Price, name="Oil Price"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.USDX_Price, name="USDX Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Oil Prices Versus USDX Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Oil</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>USDX</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)

#%% # 分析油
merge_df.corr(method='pearson')
corr_df = merge_df[["Oil_Price", "USDX_Price", "EURUSD_Price", "XAUEUR / XAUUSD Price", "XAUUSD_XAUEUR_Diff_Price"]]

# Correlation Heatmap
corrmat = corr_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
ax.set_title("Correlation Heatmap")

sns.heatmap(corrmat, square=True, annot=True)
plt.show()

# **见解和发现**
# 与黄金相比，石油与欧元或美元之间的关系乍一看没有太多的反常之处，还需要讨论。
# 石油通常与美元呈负相关关系，并与欧元/美元一起移动。由于石油是以美元定价的，而美国又是石油的净进口国，因此很容易看出这两者之间的负相关关系是如何建立的。
# 当然，石油在其他货币之间也有有趣的相关性。然而，这将在进一步的EDA中讨论，深入探讨影响外汇市场的特定商品/市场/资产。


#%% Gold / Oil Ratio
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Gold_Price, name="Gold Price"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Oil_Price, name="Oil Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Gold Prices Versus Oil Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Gold</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>Oil</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)

#%%
merge_df.corr(method='pearson')

# Adjust Correlation Dataframe
corr_df = merge_df[["Gold_Price", "Oil_Price"]]

# Correlation Heatmap
corrmat = corr_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
ax.set_title("Gold and Oil Prices Correlation Heatmap")

sns.heatmap(corrmat, square=True, annot=True)
plt.show()

#%%
# Overview of Data
sns.displot(merge_df['Gold/Oil'])
plt.show()

print("Skewness: %f" % merge_df['Gold/Oil'].skew())
print("Kurtosis: %f" % merge_df['Gold/Oil'].kurt())
print(merge_df['Gold/Oil'].describe())

#%%
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df["Gold/Oil"], name="Gold/Oil"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.US_Interest_Rates_Value, name="US Interest Rate"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Gold/Oil Prices Versus US Interest Rates"
)


# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Gold/Oil</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>US Interest Rates</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df["Gold/Oil"], name="Gold/Oil"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.EURUSD_Price, name="EURUSD Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Gold/Oil Prices Versus EURUSD Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Gold/Oil</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>EURUSD</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df["Gold/Oil"], name="Gold/Oil"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Gold_Price, name="Gold Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Gold/Oil Prices Versus Gold Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Gold/Oil</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>Gold</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df["Gold/Oil"], name="Gold/Oil"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.Oil_Price, name="Oil Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Gold/Oil Prices Versus Oil Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Gold/Oil</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>Gold</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df["Gold/Oil"], name="Gold/Oil"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.USDX_Price, name="USDX Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Gold/Oil Prices Versus USDX Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Gold/Oil</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>USDX</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)


#%%
merge_df.corr(method='pearson')

# Adjust Correlation Dataframe
corr_df = merge_df[["Gold/Oil", "US_Interest_Rates_Value", "EURUSD_Price", "USDX_Price", "Gold_Price", "Oil_Price"]]

# Correlation Heatmap
corrmat = corr_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True, annot=True)
plt.show()

# 虽然我们的其他数据与黄金/石油价格似乎没有太多的密切关联，但黄金/石油价格仍然是一个重要的因素，值得关注。
# 黄金和石油通常被认为是与美元反向的。黄金在危机中起到了 "避风港 "的投资作用，而石油与美元的反向关系源于它是以美元定价的，当美元上涨时，购买一桶石油所需的美元就会减少。
# 注意到它们与美元的反向关系的明显区别，黄金/石油比率使我们能够确定美元价格变动的具体原因/事件。

#%%
# 风险偏好
# 在参与市场时，风险管理是必不可少的。许多参与者寻求增加收益，同时试图减少/限制通常带来的下行风险的增加。这通常是以分散投资和选择 "安全 "和波动较小的证券的形式出现。
# 虽然风险管理对于保护个人资产至关重要，但全球风险偏好确实对外汇市场产生了重大影响，无论是直接还是间接。在本节中，我们将探讨如何衡量投资者的风险偏好并分析其对外汇市场的影响**。
# 波动率指数、股票指数和美元/日元
daily_usdjpy_df["Date"] = modify_datetime(daily_usdjpy_df)
# vix_df["Date"] = modify_datetime(vix_df)
snp_500_df["Date"] = modify_datetime(snp_500_df)

daily_usdjpy_df.rename(columns = {'Close' : 'USDJPY_Price', "Open": "USDJPY_Open", "High":"USDJPY_High", "Low":"USDJPY_Low", "Change %":"USDJPY_Change %"}, inplace=True)
# vix_df.rename(columns = {'Price' : 'VIX_Price', "Open": "VIX_Open", "High":"VIX_High", "Low":"VIX_Low", "Change %":"VIX_Change %"}, inplace=True)
snp_500_df.rename(columns = {'Close' : 'S&P500_Price', "Open": "S&P500_Open", "High":"S&P500_High", "Low":"S&P500_Low", "Change %":"S&P500_Change %"}, inplace=True)

merge_df = pd.merge(merge_df, daily_usdjpy_df, how="left", on="Date")
# merge_df = pd.merge(merge_df, vix_df, how="left", on="Date")
merge_df = pd.merge(merge_df, snp_500_df, how="left", on="Date")
merge_df["S&P500_Price"] = merge_df["S&P500_Price"].astype(str)
# merge_df["S&P500_Price"] = remove_comma(merge_df, "S&P500_Price")
merge_df["S&P500_Price"] = merge_df["S&P500_Price"].astype(float)

#%%
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df["S&P500_Price"], name="S&P500 Price"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=merge_df.Date, y=merge_df.USDJPY_Price, name="USDJPY Price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="S&P500 Prices Versus USDJPY Prices"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>S&P500</b> Prices", secondary_y=False)
fig.update_yaxes(title_text="<b>USDJPY</b> Prices", secondary_y=True)

fig.show()
mypltly.plot_on_webpage(fig)

#%%
merge_df.corr(method='pearson')

# Adjust Correlation Dataframe
corr_df = merge_df[["S&P500_Price",  "USDJPY_Price"]]

# Correlation Heatmap
corrmat = corr_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
ax.set_title("S&P500 / USDJPY Prices Correlation Heatmap")

sns.heatmap(corrmat, square=True, annot=True)
plt.show()











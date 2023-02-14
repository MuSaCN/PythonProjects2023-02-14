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

# %% ###### 输入参数部分 ######
''' # 输出内容保存到"工作---MT5策略研究"目录，以及MT5的Common目录。 '''
import warnings
warnings.filterwarnings('ignore')

from MyPackage.MyProjects.MT5推进分析.ForwardParse import MyClass_ForwardParse
FwdParse = MyClass_ForwardParse()

FwdParse.expertfile = "a1.箱体回调策略.ex5" # (***)基础EA(***)
FwdParse.contentfolder = r"F:\BaiduNetdiskWorkspace\工作---MT5策略研究\7.箱体回调策略" # 输出的总目录******
# (***)根据基础EA源码的Input变量的顺序来整理下面参数名(***)
FwdParse.ea_inputparalist = ["MaxBoxPeriod", "OsciBoxPeriod", "K_TrendBuyU", "K_TrendBuyD",
                             "TrendGap","K_OsciBuyLevel", "OsciGap", "CloseBuyLevel"]


# ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "USDCAD", "AUDUSD", "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY", "GBPJPY", "CHFJPY", "EURGBP", "EURAUD", "EURCHF", "EURJPY", "EURNZD", "EURCAD", "GBPCHF", "USDSGD", "CADCHF", "CADJPY", "GBPAUD", "GBPCAD", "GBPNZD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "XAUUSD", "XAGUSD"]
FwdParse.symbol = "USDJPY" # ******
FwdParse.timeframe = "TIMEFRAME_M15" # ******
FwdParse.starttime = "2015.01.01" # 推进分析数据的开始时间******
FwdParse.endtime = "2023.01.01" # 推进分析数据的结束时间(最后一个格子只做优化，不做推进)******
FwdParse.length_year = 2 # 1,2 # 样本总时间包括训练集和测试集，单位年(允许小数)******
FwdParse.step_months = 6 # 3,6 # 推进步长，单位月(允许大于12)******

# (***)优化词缀(***): -1 Complete, 0 Balance max, 6 Custom max, 7 Complex Criterion max.
FwdParse.optcriterionaffix = myMT5run.get_optcriterion_affix(optcriterion=0)

FwdParse.prepare()
FwdParse.get_timedf_matchlist_and_violent()

#%% ### 单独一次筛选 ###

# "净利润" "myCriterion" "总交易" "多头交易" "空头交易" "%总胜率" "%多胜率" "%空胜率" "TB" "Sharpe_MT5"
# "SQN_MT5_No" "Sharpe_Balance"	"SQN_Balance" "SQN_Balance_No" "Sharpe_Price" "SQN_Price" "SQN_Price_No"
# "平均盈利" "平均亏损" "盈亏比" "利润因子" "恢复因子" "期望利润" "Kelly占用仓位杠杆" "Kelly止损仓位比率"
# "Vince止损仓位比率" "最小净值" "%最大相对回撤比" "最大相对回撤比占额" "%最小保证金" "最大绝对回撤值"
# "%最大绝对回撤值占比" "回归系数" "回归截距" "LRCorrelation" "LRStandardError" "盈利总和" "亏损总和"
# "AHPR" "GHPR" "%无仓GHPR_Profit" "%无仓GHPR_Loss" "盈利交易数量" "亏损交易数量" "(int)最长获利序列"
# "最长获利序列额($)" "(int)最长亏损序列" "最长亏损序列额($)" "最大的连利($)" "(int)最大的连利序列数"
# "最大的连亏($)" "(int)最大的连亏序列数" "平均连胜序列" "平均连亏序列" "获利交易中的最大值"
# "亏损交易中的最大值"
['AHPR', 'GHPR', 'Kelly占用仓位杠杆', 'Kelly止损仓位比率', 'LRCorrelation', 'LRStandardError',
 'Sharpe_MT5', 'SQN_MT5_No', 'Sharpe_Balance', 'SQN_Balance', 'SQN_Balance_No', 'Sharpe_Price',
 'SQN_Price', 'SQN_Price_No', 'TB', 'Vince止损仓位比率',
 '多头交易', '%多胜率', '恢复因子', '回归系数', '回归截距', '获利交易中的最大值', '净利润',
 '空头交易', '%空胜率', '亏损总和', '亏损交易数量', '亏损交易中的最大值', '利润因子',
 'myCriterion', '平均盈利', '平均亏损', '平均连胜序列', '平均连亏序列',
 '期望利润', '%无仓GHPR_Profit', '%无仓GHPR_Loss', '盈亏比', '盈利总和', '盈利交易数量',
 '总交易', '%总胜率', '最小净值', '%最大相对回撤比', '最大相对回撤比占额', '%最小保证金',
 '最大绝对回撤值', '%最大绝对回撤值占比', '(int)最长获利序列', '最长获利序列额($)',
 '(int)最长亏损序列', '最长亏损序列额($)', '最大的连利($)', '(int)最大的连利序列数',
 '最大的连亏($)', '(int)最大的连亏序列数']
# ---训练集根据sortby降序排序后，从中选择count个行，再根据chooseby选择前n个最大值，再根据resultby表示结果.
sortby = "回归截距" # "Kelly占用仓位杠杆" "myCriterion" "盈亏比" "平均盈利" "盈利总和" "盈利交易数量"
count = 0.5  # 0.5一半，-1全部。注意有时候遗传算法导致结果太少，所以用-1更好
chooseby = "盈亏比" # "TB"
n = 5
resultlist=["TB", "净利润"]

FwdParse.parse_parameters(sortby=sortby, count=count, chooseby=chooseby, n=n, resultlist=resultlist)



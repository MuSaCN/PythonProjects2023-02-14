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


#%%
'''
策略优化结果使用遗传算法！
//---初始单净利润<0时 且 没有纠错单时，执行纠错
出场逻辑用的是：
  if(Class_HighestLowest.Close_Break_LastUpDown(type, symbol, MainTF, 0))
  //而不是 if(Class_HighestLowest.Close_Break_LastMiddleUpDown(2.0, type, symbol, MainTF, 0))
'''
import warnings
warnings.filterwarnings('ignore')
myDefault.set_backend_default("agg")  # 设置图片输出方式，这句必须放到类下面.
plt.show()



#%% ###### 策略参数 ######
from MyPackage.MyProjects.MT5推进分析.ForwardOpt import MyClass_ForwardOpt, myMT5run
FwdOpt = MyClass_ForwardOpt()

# ====== 策略参数 ======
# ------通用分析套件参数------
# 不需要每个参数都指定，用之前把MT5对应的EA参数默认化一下就行，需要修改的专门指定就行.
# 使用时要修改，请标注 *******
def common_set():
    myMT5run.input_set("FrameMode", "2") # 0-None 1-BTMoreResult 2-OptResult 3-ToDesk 4-GUI

# ------策略参数------
def strategy_set():
    myMT5run.input_set("MaxBoxPeriod", "68||50||1||70||Y")
    myMT5run.input_set("OsciBoxPeriod", "7||5||1||15||Y")
    myMT5run.input_set("K_TrendBuyU", "0.84||1.0||-0.02||0.7||Y")
    myMT5run.input_set("K_TrendBuyD", "0.48||0.4||0.02||0.6||Y")
    myMT5run.input_set("TrendGap", "500||0||100||1000||Y")
    myMT5run.input_set("K_OsciBuyLevel", "0.15||0.05||0.05||0.2||Y")
    myMT5run.input_set("OsciGap", "120||80||10||150||Y")
    myMT5run.input_set("CloseBuyLevel", "0.82||0.8||0.02||0.9||Y")


#%% ###### 策略优化 ######

FwdOpt.experfolder = "My_Experts\\Strategy深度研究\\4.箱体回调策略"  # (***)基础EA所在的目录(***)
FwdOpt.expertfile = "a1.箱体回调策略.ex5"  # (***)基础EA(***)
FwdOpt.contentfolder = r"F:\BaiduNetdiskWorkspace\工作---MT5策略研究\7.箱体回调策略" # 输出的总目录******

# 推进测试的起止时间
FwdOpt.starttime = "2015.01.01" # ************
FwdOpt.endtime = "2023.01.01" # ************
FwdOpt.step_months = 6 # 6, 3 # 推进步长，单位月 # ************
FwdOpt.length_year = 2 # 2, 1 # 样本总时间包括训练集和测试集 # ************

FwdOpt.symbollist = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "USDCAD", "AUDUSD", "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY", "GBPJPY", "CHFJPY", "EURGBP", "EURAUD", "EURCHF", "EURJPY", "EURNZD", "EURCAD", "GBPCHF", "USDSGD", "CADCHF", "CADJPY", "GBPAUD", "GBPCAD", "GBPNZD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "XAUUSD", "XAGUSD"] # *********



FwdOpt.timeframe = "TIMEFRAME_M15"  # ************

FwdOpt.forwardmode = 4  # *** 向前检测 (0 "No", 1 "1/2", 2 "1/3", 3 "1/4", 4 "Custom")
FwdOpt.model = 1  # *** 0 "每笔分时", 1 "1 分钟 OHLC", 2 "仅开盘价", 3 "数学计算", 4 "每个点基于实时点"
FwdOpt.optimization = 2  # *** 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"


#%%
FwdOpt.prepare(common_set, strategy_set)

#---测试下哪个优化标准更能找到好策略
# -1 -- Complete, 0 -- Balance max, 1 -- Profit Factor max, 2 -- Expected Payoff max, 3 -- Drawdown min, 4 -- Recovery Factor max, 5 -- Sharpe Ratio max, 6 -- Custom max, 7 -- Complex Criterion max.
print("run0")
FwdOpt.run(0)

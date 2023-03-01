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
''' 需要有对应的EA文件，比如 a4.f5.EURUSD.M15.ex5，且要写好！ '''
import warnings
warnings.filterwarnings('ignore')
from MyPackage.MyProjects.MT5推进分析.ForwardRepairAdd import MyClass_ForwardRepairAdd, myMT5run
FwdRprAd = MyClass_ForwardRepairAdd()

# (***)推进回测(***)
FwdRprAd.timeframe = "TIMEFRAME_M15" # 策略的时间框******
FwdRprAd.bt_starttime = "2016.07.01"  # 手动指定******，一般为推进样本外的起始
FwdRprAd.bt_endtime = "2023.02.08"  # 手动指定******，一般为最近的时间

# FwdRprAd.bt_starttime = "2019.02.01"  # 敏感特殊时间测试！！！
# FwdRprAd.bt_endtime = "2019.06.20"  # 敏感特殊时间测试！！！

# (***)输出目录(***)
# 输出的总目录******
FwdRprAd.contentfolder = r"F:\BaiduNetdiskWorkspace\工作---MT5策略研究\8.ZigZag与均线缠绕后突破轨道"
# 之前手工建立的目录******

FwdRprAd.bt_folder = FwdRprAd.contentfolder + r"\4.组合1.2016-07-01.2023-01-01.IC"
# FwdRprAd.bt_folder = FwdRprAd.contentfolder + r"\4.组合.2016-07-01.2023-01-01.IC\敏感品种敏感时段"


# (***)推进回测EA的目录(后面不能带\\)和文件名(***)
FwdRprAd.bt_experfolder = "My_Experts\\Strategy深度研究\\5.ZigZag与均线缠绕后突破轨道\\推进交易.2Y6M\\组合"
# 敏感品种敏感时段测试
# FwdRprAd.bt_experfolder = "My_Experts\\Strategy深度研究\\5.ZigZag与均线缠绕后突破轨道\\推进交易.2Y6M"

# (***)ex5的名称格式(***)，要修改
FwdRprAd.bt_expertnameform = "a5.f5.组合.{}.ex5" # 必须是 a5.f5.组合.M15 格式，最后{}对应时间框词缀.
# 敏感品种敏感时段测试
# FwdRprAd.bt_expertnameform = "a4.f5.AUDJPY.{}.ex5" # 必须是 a5.f5.组合.M15 格式，最后{}对应时间框词缀.


# (***)回测的设置(***)，一般只要修改 delays
FwdRprAd.bt_model = 1  # 0 "每笔分时", 1 "1 分钟 OHLC", 2 "仅开盘价", 3 "数学计算", 4 "每个点基于实时点"
FwdRprAd.bt_profitinpips = 0 # profitinpips = 1 用pips作为利润，不用具体的货币。0用具体货币，且考虑佣金


#%%
# ###### 单次回测主要函数 ######
# ------通用分析套件参数------
def common_set():
    myMT5run.input_set("FrameMode", "2")  # 0-None 1-BTMoreResult 2-OptResult 3-ToDesk 4-GUI
    myMT5run.input_set("Inp_CustomMode", "24")  # 24-MarginMin, 25-MaxRelativeDDPct


#%% ### SplitFund
# (***)不同模式不同保存目录(***)
FwdRprAd.bt_reportfolder = FwdRprAd.bt_folder + "\\" + "SplitFund"
def strategy_set1(): # SplitFund
    myMT5run.input_set("Inp_MM_Mode", "3||0||0||8||N") # SplitFund
    myMT5run.input_set("Inp_Lots_IncreDelta", "100||100||50||2000||Y")
    myMT5run.input_set("Inp_Lots_IncreInitLots", "0.01||0.1||0.010000||1.000000||N")
FwdRprAd.prepare(common_set, strategy_set1)
FwdRprAd.combine_symbol_opt(symbol="AUDJPY", optimization=1, deposit=2000, shutdownterminal=1)

#%% ### SplitFormula
# (***)不同模式不同保存目录(***)
FwdRprAd.bt_reportfolder = FwdRprAd.bt_folder + "\\" + "SplitFormula"
def strategy_set2(): # SplitFormula
    myMT5run.input_set("Inp_MM_Mode", "4||0||0||8||N") # SplitFormula
    myMT5run.input_set("Inp_Lots_IncreDelta", "100||100||50||2000||Y")
    myMT5run.input_set("Inp_Lots_IncreInitLots", "0.01||0.1||0.010000||1.000000||N")
FwdRprAd.prepare(common_set, strategy_set2)
FwdRprAd.combine_symbol_opt(symbol="AUDJPY", optimization=1, deposit=2000, shutdownterminal=1)

#%% ### SplitFormula
# (***)不同模式不同保存目录(***)
FwdRprAd.bt_reportfolder = FwdRprAd.bt_folder + "\\" + "StepBalanceRatio"
def strategy_set3(): # StepBalanceRatio
    myMT5run.input_set("Inp_MM_Mode", "8||0||0||8||N") # SplitFormula
    myMT5run.input_set("Inp_Lots_BasicEveryLot", "200000||5000.0||500.000000||50000.000000||N")
    myMT5run.input_set("Inp_Lots_BasicStep", "2000||500||100||2000||Y")
FwdRprAd.prepare(common_set, strategy_set3)
FwdRprAd.combine_symbol_opt(symbol="AUDJPY", optimization=1, deposit=2000, shutdownterminal=1)

#%% ### OccupyMarginPct
# (***)不同模式不同保存目录(***)
FwdRprAd.bt_reportfolder = FwdRprAd.bt_folder + "\\" + "OccupyMarginPct"
def strategy_set4(): # StepBalanceRatio
    myMT5run.input_set("Inp_MM_Mode", "7||0||0||8||N") # OccupyMarginPct
    myMT5run.input_set("Inp_Lots_SLRiskPercent", "0.01||0.005||0.001||0.02||Y")
FwdRprAd.prepare(common_set, strategy_set4)
FwdRprAd.combine_symbol_opt(symbol="AUDJPY", optimization=1, deposit=2000, shutdownterminal=1)



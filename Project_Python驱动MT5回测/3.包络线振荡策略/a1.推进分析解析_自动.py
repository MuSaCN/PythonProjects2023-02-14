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
''' # 输出内容保存到"工作---MT5策略研究"目录，以及MT5的Common目录。 '''
import warnings
warnings.filterwarnings('ignore')
myDefault.set_backend_default("agg")  # 设置图片输出方式，这句必须放到类下面.
plt.show()


#%%
from MyPackage.MyProjects.MT5推进分析.ForwardAutoParse import MyClass_ForwardAutoParse, myMT5run
FwdAuto = MyClass_ForwardAutoParse()


# (***)基础EA(***)。用于优化分析的，注意不同于下面推进回测的EA，后者要阶段更新参数。
FwdAuto.expertfile = "a1.包络线振荡策略.ex5"
FwdAuto.contentfolder = r"F:\BaiduNetdiskWorkspace\工作---MT5策略研究\6.包络线振荡策略" # 输出的总目录******
# (***)根据基础EA源码的Input变量的顺序来整理下面参数名(***)
FwdAuto.ea_inputparalist = ["Inp_SigMode", "Inp_Ma_Period", "Inp_Ma_Method", "Inp_Applied_Price", "Inp_Deviation","Inp_SLMuiltple", "Inp_Filter0", "Inp_Filter1"]

FwdAuto.symbollist = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "USDCAD", "AUDUSD", "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY", "GBPJPY", "CHFJPY", "EURGBP", "EURAUD", "EURCHF", "EURJPY", "EURNZD", "EURCAD", "GBPCHF", "USDSGD", "CADCHF", "CADJPY", "GBPAUD", "GBPCAD", "GBPNZD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "XAUUSD", "XAGUSD"] # *********


FwdAuto.timeframe = "TIMEFRAME_M30" # 策略的时间框******


# (******)MT5上的推进交易EA读取文档也要修改(******)，时间要调整为forward_starttime,forward_endtime
# (***)推进分析的相关参数(***)
FwdAuto.forward_starttime = "2015.01.01" # 推进分析数据的开始时间******
FwdAuto.forward_endtime = "2023.01.01" # 推进分析数据的结束时间(最后一个格子只做优化，不做推进)******
FwdAuto.length_year = 2 # 1,2 # 样本总时间包括训练集和测试集，单位年(允许小数)******
FwdAuto.step_months = 6 # 3,6 # 推进步长，单位月(允许大于12)******


# (***)优化词缀(***): -1 Complete, 0 Balance max, 6 Custom max, 7 Complex Criterion max.
FwdAuto.optcriterionaffix = myMT5run.get_optcriterion_affix(optcriterion=0) # ******


# (***)推进回测EA的目录(后面不能带\\)和文件名(***)
FwdAuto.bt_experfolder = "My_Experts\\Strategy深度研究\\3.包络线振荡策略\\推进交易.2Y6M"
FwdAuto.bt_expertfile = "a1.f3._Symbol.{}.ex5".format(myMT5run.timeframe_to_ini_affix(FwdAuto.timeframe))
# (***)推进回测的时间起始(***)
FwdAuto.bt_starttime = "2016.07.01"  # 手动指定******
FwdAuto.bt_endtime = "2023.01.01"  # 手动指定******


# 推进回测保存的总目录
FwdAuto.bt_folder = FwdAuto.contentfolder + r"\1.推进回测.{}.{}.{}".format(
    FwdAuto.optcriterionaffix,
    myMT5run.change_timestr_format(FwdAuto.bt_starttime),
    myMT5run.change_timestr_format(FwdAuto.bt_endtime))


#%% ###### 主要函数 ######

# ------通用分析套件参数------
# 不需要每个参数都指定，用之前把MT5对应的EA参数默认化一下就行，需要修改的专门指定就行.
# 使用时要修改，请标注 *******
def common_set():
    myMT5run.input_set("FrameMode", "2") # 0-None 1-BTMoreResult 2-OptResult 3-ToDesk 4-GUI

# ---(***)推进回测策略参数(***)---
def strategy_set():
    pass


#%%
FwdAuto.prepare(common_set, strategy_set)
FwdAuto.run()



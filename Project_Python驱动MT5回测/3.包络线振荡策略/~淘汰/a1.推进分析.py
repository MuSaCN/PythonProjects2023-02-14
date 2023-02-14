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
策略优化结果不多，不使用遗传算法！
'''
import warnings
warnings.filterwarnings('ignore')
myDefault.set_backend_default("agg")  # 设置图片输出方式，这句必须放到类下面.
plt.show()



#%% ###### 策略参数 ######

# ====== 策略参数 ======
# ------通用分析套件参数(版本2022.10.15)------
# 使用时要修改，请标注 *******
def common_set():
    myMT5run.input_set("FrameMode", "2") # 0-FRAME_None 1-BTMoreResult 2-OptResult

# ------策略参数------
def strategy_set():
    myMT5run.input_set("Inp_SigMode", "1||1||1||2||Y")  # 1-左侧入场，2-右侧入场。
    myMT5run.input_set("Inp_Ma_Period", "20||20||1||40||Y") # ************
    myMT5run.input_set("Inp_Ma_Method", "0||0||0||3||N") # ************
    myMT5run.input_set("Inp_Applied_Price", "1||1||0||7||N") # ************
    myMT5run.input_set("Inp_Deviation", "0.1||0.1||0.05||0.7||Y") # ************
    myMT5run.input_set("Inp_SLMuiltple", "2||2.0||0.200000||20.000000||N")  # 初始止损的倍数
    myMT5run.input_set("Inp_Filter0", "true||false||0||true||N")  # 信号过滤0：前一单做多亏，则当前只能做空；前一单做空亏，则当前只能做多。
    myMT5run.input_set("Inp_Filter1", "true||false||0||true||Y") # 信号过滤1：D1上过滤震荡，D1上震荡才允许进场。


#%% ###### 策略优化 ######

experfolder = "My_Experts\\Strategy深度研究\\3.包络线振荡策略"  # (***)基础EA所在的目录(***)
expertfile = "a1.包络线振荡策略.ex5"  # (***)基础EA(***)
contentfolder = r"F:\BaiduNetdiskWorkspace\工作---MT5策略研究\6.包络线振荡策略" # 输出的总目录******

# 推进测试的起止时间
starttime = "2015.01.01" # ************
endtime = "2023.01.1" # ************
step_months = 6 # 6, 3 # 推进步长，单位月 # ************
length_year = 2 # 2, 1 # 样本总时间包括训练集和测试集 # ************

symbollist = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF", "XAUUSD", "XAGUSD", "AUDJPY","CHFJPY","EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","GBPAUD","GBPCAD","GBPCHF","GBPJPY","NZDJPY"] # *********
timeframe = "TIMEFRAME_M30"  # ************

forwardmode = 4  # *** 向前检测 (0 "No", 1 "1/2", 2 "1/3", 3 "1/4", 4 "Custom")
model = 1  # *** 0 "每笔分时", 1 "1 分钟 OHLC", 2 "仅开盘价", 3 "数学计算", 4 "每个点基于实时点"
optimization = 2  # *** 0 禁用优化, 1 "慢速完整算法", 2 "快速遗传算法", 3 "所有市场观察里选择的品种"


#%% 清空下Common目录Files目录下已经输出过的csv文件
commonfile_folder = __mypath__.get_mt5_terminal_path() + r"\Common\Files"
commonfile = commonfile_folder + "\\" + expertfile.rsplit(".", 1)[0] + ".csv"
myfile.remove_dir_or_file(commonfile)


#%%
timeaffix0 = myMT5run.change_timestr_format(starttime)
timeaffix1 = myMT5run.change_timestr_format(endtime)
starttime = pd.Timestamp(starttime)
endtime = pd.Timestamp(endtime)

timedf = myMT5run.get_everystep_time(starttime, endtime, step_months=step_months, length_year=length_year)

#%%
#---测试下哪个优化标准更能找到好策略
# -1 -- Complete, 0 -- Balance max, 1 -- Profit Factor max, 2 -- Expected Payoff max, 3 -- Drawdown min, 4 -- Recovery Factor max, 5 -- Sharpe Ratio max, 6 -- Custom max, 7 -- Complex Criterion max.
def run(criterionindex=0):
    optcriterionaffix = myMT5run.get_optcriterion_affix(optcriterion=criterionindex)  # ***
    optcriterion = criterionindex # *** 0 Balance max, 1 Profit Factor max, 2 Expected Payoff max, 3 Drawdown min, 4 Recovery Factor max, 5 Sharpe Ratio max, 6 Custom max, 7 Complex Criterion max


    # ---
    for symbol in symbollist:
        if symbol in []:  # symbol = "EURUSD" "GBPUSD"
            continue

        length = "%sY" % length_year
        step = "%sM" % step_months

        reportfolder = contentfolder + r"\推进分析.{}\推进.{}.{}.length={}.step={}".\
            format(optcriterionaffix, symbol,myMT5run.timeframe_to_ini_affix(timeframe),
                   length, step)  # 以 "推进.EURUSD.M30.length=2Y.step=6M" 格式
        expertname = experfolder + "\\" + expertfile

        for i, row in timedf.iterrows():
            # 时间参数必须转成"%Y.%m.%d"字符串
            fromdate = row["from"]
            forwarddate = row["forward"]
            todate = row["to"]
            print("======开始测试：fromdate={}, forwarddate={}, todate={}".format(fromdate, forwarddate, todate))

            # ---最后一步要调整下t1和t2
            islast = pd.Timestamp(forwarddate) == pd.Timestamp(endtime)
            tf_affix = myMT5run.timeframe_to_ini_affix(timeframe)
            t0 = myMT5run.change_timestr_format(fromdate)
            t1 = myMT5run.change_timestr_format(forwarddate) if islast is False else None
            t2 = myMT5run.change_timestr_format(todate) if islast is False else myMT5run.change_timestr_format(forwarddate)

            # ---xml格式优化报告的目录
            reportfile = reportfolder + "\\{}.{}.{}.{}.{}.{}.Crit={}.xml".format(
                expertfile.rsplit(sep=".", maxsplit=1)[0], symbol, tf_affix, t0, t1, t2, optcriterion)
            print("reportfile=", reportfile)

            # 如果t1是None表示不是向前分析
            iforwardmode = 0 if t1 is None else forwardmode # 向前检测 (0 "No", 1 "1/2", 2 "1/3", 3 "1/4", 4 "Custom")
            todate = forwarddate if t1 is None else todate
            print("t0={} t1={} t2={}".format(t0, t1, t2))
            print("fromdate={} forwarddate={} todate={}".format(fromdate, forwarddate, todate))
            print("forwardmode={} ".format(iforwardmode))

            # 检测文件是否存在，存在则不需要再次优化
            csvfile = reportfolder + "\\{}.{}.{}.{}.{}.{}.csv".format(expertfile.rsplit(sep=".", maxsplit=1)[0], symbol, tf_affix, t0, t1, t2)
            if __mypath__.path_exists(reportfile) and __mypath__.path_exists(csvfile):
                print("已经完成：", reportfile)
                continue


            myMT5run.__init__()
            myMT5run.config_Tester(expertname, symbol, timeframe, fromdate=fromdate, todate=todate,
                                   forwardmode=iforwardmode, forwarddate=forwarddate,
                                   delays=0, model=model, optimization=optimization,
                                   optcriterion=optcriterion, reportfile=reportfile)
            common_set()
            strategy_set()
            # ---检查参数输入是否匹配优化的模式，且写出配置结果。
            myMT5run.check_inputs_and_write()
            myMT5run.run_MT5()


#%% 测试下哪个优化标准更能找到好策略
print("run0")
run(0)
# print("run6")
# run(6)
# print("run7")
# run(7)





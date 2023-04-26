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
mypltly = MyPlot.MyClass_Plotly()  # plotly画图相关
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
# 对于稳定的交易，通常建议将交易工具和交易策略多样化。机器学习模型也是如此：创建几个较简单的模型比创建一个复杂的模型要容易。但要将这些模型组合成一个ONNX模型可能会很困难。
# 然而，有可能在一个MQL5程序中结合几个经过训练的ONNX模型。在这篇文章中，我们将考虑其中一个称为投票分类器的组合。我们将向您展示实现这样一个集合体是多么容易。

# 项目的模型
# 对于我们的例子，我们将使用两个简单的模型：一个回归价格预测模型和一个分类价格运动预测模型。这两个模型的主要区别是，回归预测的是数量，而分类预测的是类别。
# 第一个模型是回归模型。
# 它使用2003年至2022年底的欧元兑美元D1数据进行训练。训练是使用10个OHLC价格系列进行的。为了提高模型的可训练性，我们将价格标准化，用系列中的平均价格除以系列中的标准差。因此，我们把一个系列放到一定的范围内，均值为0，差值为1，这样可以提高训练期间的收敛性。
# 结果是，该模型应该预测第二天的收盘价。
# 这个模型非常简单。这里提供的只是演示用的。

from datetime import datetime
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
import tf2onnx
from sklearn.model_selection import train_test_split
from tqdm import tqdm # tqdm是Python进度条库,可以在 Python长循环中添加一个进度提示信息。

if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# we will save generated onnx-file near the our script
# data_path=argv[0]
# last_index=data_path.rfind("\\")+1
# data_path=data_path[0:last_index]
data_path = __mypath__.get_desktop_path() + "\\"
print("data path to save onnx model",data_path)

# input parameters
inp_model_name = "model.eurusd.D1.10.onnx"
inp_history_size = 10
inp_start_date = datetime(2003, 1, 1, 0)
inp_end_date = datetime(2023, 1, 1, 0)

# get data from client terminal
# eurusd_rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_D1, inp_start_date, inp_end_date)
# df = pd.DataFrame(eurusd_rates)
df = myMT5Pro.getsymboldata("EURUSD", "TIMEFRAME_D1", inp_start_date, inp_end_date, index_time=False,col_capitalize=False)


# collect dataset subroutine
def collect_dataset(df: pd.DataFrame, history_size: int):
    """
    为以下回归问题收集数据集:
    - input: history_size consecutive H1 bars;
    - output: close price for the next bar.

    :param df: D1 bars for a range of dates
    :param history_size: how many bars should be considered for making a prediction
    :return: features and labels
    """
    n = len(df)
    xs = []
    ys = []
    # 这里的输入以时间序列的形式，而不是把特征值全部放入1行。
    # 每组的x为history_size行，y为1个
    for i in tqdm(range(n - history_size)): # i=0
        # 切片大小为 history_size+1
        w = df.iloc[i: i + history_size + 1]
        # x为前10个，y为第11个
        x = w[['open', 'high', 'low', 'close']].iloc[:-1].values
        y = w.iloc[-1]['close']
        xs.append(x)
        ys.append(y)

    X = np.array(xs)
    y = np.array(ys)
    return X, y
###

# get prices
X, y = collect_dataset(df, history_size=inp_history_size)
X.shape # (5185, 10, 4)

# normalize prices
m = X.mean(axis=1, keepdims=True) # axis=1表示以中间的那个维度
s = X.std(axis=1, keepdims=True)
m.shape # (5185, 1, 4)
X_norm = (X - m) / s
X_norm.shape # (5185, 10, 4)
y.shape # (5185,)
y_norm = (y - m[:, 0, 3]) / s[:, 0, 3] # 索引3表示close，与y的意义匹配
y_norm.shape # (5185,)

# split data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=0)

# define model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(inp_history_size, 4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# model training for 50 epochs
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001)
history = model.fit(X_train, y_train, epochs=50, verbose=2, validation_split=0.15, callbacks=[lr_reduction])

# model evaluation
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"test_loss={test_loss:.3f}")
print(f"test_mae={test_mae:.3f}")

# save model to onnx
output_path = data_path+inp_model_name
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
print(f"saved model to {output_path}")

# finish
mt5.shutdown()

#%%
# 假设我们的回归模型被执行，得出的预测价格应该被转化为以下类别：价格下降，价格不变，价格上升。这是为了组织投票分类器而需要的。

# 第二个模型是分类模型。
# 它是在2010年至2022年底的欧元兑美元D1上训练的。训练是使用一系列的63个收盘价进行的。在输出端必须定义三个类别之一：价格将下降，价格将保持在10点以内，或者价格将上升。正是因为第二类，我们不得不使用2010年以来的数据来训练模型--在此之前，在2009年，市场从4位数转换为5位数的准确性。因此，一个旧点变成了十个新点。
# 和以前的模型一样，价格被归一化。归一化是相同的：我们用系列中的平均价格的偏差除以系列中的标准偏差。这个模型的想法在《用Keras中的MLP进行金融时序预测》（俄语）一文中有所描述。这个模型也只是为了演示而设计的。
from datetime import datetime
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
import tf2onnx
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import SGD
from keras import regularizers

# initialize MetaTrader 5 client terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# we will save the generated onnx-file near the our script
# data_path=argv[0]
# last_index=data_path.rfind("\\")+1
# data_path=data_path[0:last_index]
data_path = __mypath__.get_desktop_path() + "\\"
print("data path to save onnx model",data_path)

# input parameters
inp_model_name = "model.eurusd.D1.63.onnx"
inp_history_size = 63
inp_start_date = datetime(2010, 1, 1, 0)
inp_end_date = datetime(2023, 1, 1, 0)

# get data from the client terminal
eurusd_rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_D1, inp_start_date, inp_end_date)
df = pd.DataFrame(eurusd_rates)
df = myMT5Pro.getsymboldata("EURUSD", "TIMEFRAME_D1", inp_start_date, inp_end_date, index_time=False,col_capitalize=False)


# collect dataset subroutine
def collect_dataset(df: pd.DataFrame, history_size: int):
    """
    为以下回归问题收集数据集:
    - input: history_size consecutive H1 bars;
    - output: close price for the next bar.

    :param df: H1 bars for a range of dates
    :param history_size: how many bars should be considered for making a prediction
    :return: features and labels
    """
    n = len(df)
    xs = []
    ys = []
    for i in tqdm(range(n - history_size)): # i=0
        w = df.iloc[i: i + history_size + 1]
        # x为63行
        x = w[['close']].iloc[:-1].values
        delta = x[-1] - w.iloc[-1]['close']
        # y这里以二进制形式表示分类
        if np.abs(delta)<=0.0001:
           y = 0, 1, 0
        else:
           if delta<0:
              y = 1, 0, 0
           else:
              y = 0, 0, 1

        xs.append(x)
        ys.append(y)

    X = np.array(xs)
    Y = np.array(ys)
    return X, Y
###


# get prices
X, Y = collect_dataset(df, history_size=inp_history_size)

# normalize prices
m = X.mean(axis=1, keepdims=True)
s = X.std(axis=1, keepdims=True)
X_norm = (X - m) / s

# split data to train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.1, random_state=0)

# define model
model = Sequential()
model.add(Dense(64, input_dim=inp_history_size, activity_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.3))
model.add(Dense(16, activity_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(3))
model.add(Activation('softmax'))

opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# model training for 300 epochs
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.00001)
history = model.fit(X_train, Y_train, epochs=300, validation_data=(X_test, Y_test), shuffle = True, batch_size=128, verbose=2, callbacks=[lr_reduction])

# model evaluation
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"test_loss={test_loss:.3f}")
print(f"test_accuracy={test_accuracy:.3f}")

# save model to onnx
output_path = data_path+inp_model_name
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
print(f"saved model to {output_path}")

# finish
mt5.shutdown()

# 这些模型用数据训练到2022年底，从而留下了在策略测试器中展示其操作的时间。

#%% MQL5专家顾问中的ONNX模型组合
# 下面是一个简单的专家顾问，以展示模型组合的可能性。在MQL5中使用ONNX模型的主要原则已在上一篇文章的第二部分描述。
# 前瞻性声明和定义









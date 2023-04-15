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
# 在这个笔记本中，训练了一个LSTM模型，预测欧元兑美元外汇对在第二天的收盘价是否会比5天前的收盘价高或低。
# 因为在这项研究中，我只对是否有可能预测价格方向（上涨/下跌或上涨/下跌）感兴趣，显然，我需要开始寻找一种分类方法。
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt




#%%
# 处理输入数据文件：
# 读取CSV文件，其中包含欧元：美元货币对的历史价格数据
# 对列做一些重命名
# 标记一列为索引列
# 确保所有的数据值都是浮动类型的
# 从数据集中删除所有未使用的列
# Read the CSV file
df = myMT5Pro.getsymboldata("EURUSD", "TIMEFRAME_D1", [2000,1,1,0,0,0], [2024,1,1,0,0,0], index_time=True,col_capitalize=True)
df = df[['Time','Open','Close','High','Low','Tick_volume']]

# Rename the columns
df.rename(columns={'Time' : 'timestamp', 'Open' : 'open', 'Close' : 'close', 'High' : 'high', 'Low' : 'low', 'Close' : 'close', 'Tick_volume' : 'volume'}, inplace=True)
# Set the 'timestamp' column as an index for the dataframe
df.set_index('timestamp', inplace=True)
# Convert all values as floating point numbers
# df = df.astype(float)
# Check on the data types of the columns
print('Data type of each column in Dataframe: ')
print(df.dtypes)

df = df.iloc[0:]

# Contents of the dataframe
print('Contents of the Dataframe: ')
print(df)

# Drop unused columns from dataframe
columns_to_drop = ['open', 'high', 'low', 'volume']
df.drop(columns_to_drop, axis=1, inplace=True)
print(df.head(15))

# Plotting
plt.title('EURUSD Historical Close Prices')
plt.plot(df['close'].values, label='Close')
plt.legend(loc="upper right")
plt.show()

#%% Dataset slicing
# 当把机器学习算法应用于金融时间序列时，该模型一般以一个时间窗口作为输入。例如，20个连续的收盘价。这个时间窗口所使用的价格数量被定义为回看期。
look_back = 20

# Frame a time series as a supervised learning dataset.
# Arguments:
#    data: Sequence of observations as a list or NumPy array.
#    n_in: Number of lag observations as input (X).
#    n_out: Number of observations as output (y).
#    drop_nan: Boolean whether or not to drop rows with NaN values.
# Returns:
#    Pandas DataFrame of series framed for supervised learning.
def series_to_supervised(data, n_in=1, n_out=1, drop_nan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)
    return agg

# Convert the dataframe to a series
series = df.values

# Convert the series to a dataframe suitable for supervised learning
df_supervised = series_to_supervised(data=series, n_in=look_back, n_out=1, drop_nan=True)
print(df_supervised.head(15))

#%%
# 以与滞后5期作比较来设定y
# Now we should add an additional column to this supervised dataframe which will contain either the value 1 or 0
# If('var1_t' > 'var1_t5') then 'var1(t) >= var1(t-5)' = 1 else 'var1(t) >= var1(t-5)' = 0
df_supervised.loc[df_supervised['var1(t)'] >= df_supervised['var1(t-5)'], 'var1(t) >= var1(t-5)'] = 1
df_supervised.loc[df_supervised['var1(t)'] < df_supervised['var1(t-5)'], 'var1(t) >= var1(t-5)'] = 0
print(df_supervised.head(15))

# After that we can safely remove the column var1_t
df_supervised.drop('var1(t)', axis=1, inplace=True)
print(df_supervised.head(15))


#%% Split the data into a training and test series
pct_train = 80

series_supervised = df_supervised.values
train_size = int(len(series_supervised) * pct_train / 100)
train = series_supervised[0:train_size]
test = series_supervised[train_size:]
print('Training series:')
print(train)
print(train.shape)
print('Test series:')
print(test)
print(test.shape)

#%% Split the training data into features and labels
# train_X, train_y
train_X = train[:, 0:-1]
train_y = train[:, -1]
print('Training series X:')
print(train_X)
print('Training series y:')
print(train_y)

#%% Split the test data into features and labels
# test_X, test_y
test_X = test[:, 0:-1]
test_y = test[:, -1]
print('Test series X:')
print(test_X)
print('Test series y:')
print(test_y)

#%% 对训练和测试数据的输入特征进行缩放
# 在完成了切片之后，在完整的数据集被分割成训练集和测试集之后，我们现在有了一个可以独立缩放的切片量。
# 我们期望我们的机器学习算法能够识别导致上涨或下跌的价格模式。将每个片断独立于其他片断进行缩放，可以使训练更容易，因为这样我们就可以消除由于长期市场趋势而产生的全球范围效应。
# Scale each row in train_X
train_X_scaled = np.array([])
for i in range(0, len(train_X)): # i=0
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    reshaped = train_X[i].reshape(len(train_X[i]), 1)
    scaler = scaler.fit(reshaped)
    # transform train
    scaled = scaler.transform(reshaped)
    train_X_scaled = np.append(train_X_scaled, scaled)

train_X_scaled = train_X_scaled.reshape(train_X.shape[0], train_X.shape[1])
print('Training series X scaled:')
print(train_X_scaled)
print(train_X_scaled.shape)

# Scale each row in test_X
test_X_scaled = np.array([])
for i in range(0, len(test_X)):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    reshaped = test_X[i].reshape(len(test_X[i]), 1)
    scaler = scaler.fit(reshaped)
    # transform train
    scaled = scaler.transform(reshaped)
    test_X_scaled = np.append(test_X_scaled, scaled)

test_X_scaled = test_X_scaled.reshape(test_X.shape[0], test_X.shape[1])
print('Test series X scaled:')
print(test_X_scaled)
print(test_X_scaled.shape)


#%% Build and train a LSTM model
time_steps = 20
features = 1
neurons = 64
batch_size = 1
nb_epoch = 10

# reshape input to be 3D [samples, timesteps, features]
train_X_scaled = train_X_scaled.reshape((train_X_scaled.shape[0], time_steps, features))
test_X_scaled = test_X_scaled.reshape((test_X.shape[0], time_steps, features))
print(train_X_scaled.shape, train_y.shape, test_X_scaled.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, train_X_scaled.shape[1], train_X_scaled.shape[2]), stateful=True, return_sequences=True))
model.add(Dropout(0.2))
#model.add(LSTM(neurons, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(Dropout(0.2))
model.add(LSTM(neurons, return_sequences=False))  # return a single vector of dimension 32
model.add(Dropout(0.2))
#model.add(Dense(1))
model.add(Dense(1,activation='sigmoid'))
#model.compile(loss='mae', optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#%%
# fit network
history = model.fit(train_X_scaled, train_y, epochs=nb_epoch, batch_size=1, validation_data=(test_X_scaled, test_y), verbose=1, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Forecast the entire training dataset to build up state for forecasting
predictions_on_training = model.predict(train_X_scaled, batch_size=1)
print('Expected values on training')
print(train_y)
print(train_y.shape)
print('Predicted values on training')
print(predictions_on_training)
print(predictions_on_training.shape)

plt.title('Expected and predicted values on Training Set')
plt.plot(train_y[-20:], label="expected")
plt.plot(predictions_on_training[-20:], label="predicted")
plt.legend(loc="upper left")
plt.show()

#%%
# make a one-step forecast
print(test_X_scaled)
print(test_X_scaled.shape)
print(test_y)
print(test_y.shape)
# Forecast on test data
predictions_on_test = np.array([])
print(len(test_X_scaled))
for i in range(len(test_X_scaled)):
    X = test_X_scaled[i, :]
    X = X.reshape(1, X.shape[0], X.shape[1])
    y = model.predict(X, batch_size=1)
    predictions_on_test = np.append(predictions_on_test, y)

print('Expected values on test')
print(test_y)
print(test_y.shape)
print('Predicted values on test')
print(predictions_on_test)
print(predictions_on_test.shape)

plt.title('Expected and predicted values on Test Set')
plt.plot(test_y[-20:], label="expected")
plt.plot(predictions_on_test[-20:], label="predicted")
plt.legend(loc="upper left")
plt.show()

#%%
# How many ups and downs to we have in the test set
unique, counts = np.unique(test_y, return_counts=True)
print(unique)
print(counts)

# How many ups do we have in the predictions
print((predictions_on_test > 0.5).sum())
# How many downs do we have in the predictions
print((predictions_on_test < 0.5).sum())

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
dirs = ['Up', 'Down']
count_test_y = [counts[1],counts[0]]
ax.bar(dirs ,count_test_y)
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
dirs = ['Up', 'Down']
count_predictions_on_test = [(predictions_on_test > 0.5).sum(),(predictions_on_test < 0.5).sum()]
ax.bar(dirs ,count_predictions_on_test)
plt.show()



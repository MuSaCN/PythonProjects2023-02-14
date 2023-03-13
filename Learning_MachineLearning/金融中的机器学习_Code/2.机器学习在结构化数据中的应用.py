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

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
from tqdm import tqdm

#%%
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.savefig('confusion.png')
    plt.show()

#%%

df = pd.read_csv(r'E:\学习\database数据集\金融中的机器学习\PS_20174392719_1491204439457_log.csv')
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
len(df)

#%%
df.head()

#%%
# 查看诈骗交易的交易类型 'TRANSFER', 'CASH_OUT'
df.loc[df.isFraud == 1].type.drop_duplicates().values
df.loc[df.isFraud == 1].type.unique()

#%%
# 只研究这两者类型的交易
df = df[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

#%%

len(df)

#%%
# 诈骗交易且类型为 TRANSFER 的交易类中位数
df.loc[(df.isFraud == 1) & (df.type == 'TRANSFER')].amount.median()

#%%
# 非诈骗交易且类型为 TRANSFER 的交易类中位数
df.loc[(df.isFraud == 0) & (df.type == 'TRANSFER')].amount.median()

#%%
# 使用启发式模式来预测，创建 Fraud_Heuristic 列.
# 满足条件1，否则0.
df['Fraud_Heuristic'] = np.where(((df['type'] == 'TRANSFER') & (df['amount'] > 200000)),1,0)

#%%

df['Fraud_Heuristic'].sum()

# %%
# 调和平均数处理不可相加，但是混合的事物。关键点在于混合。
# f1分数为准确率和召回率的调和平均数.
from sklearn.metrics import f1_score

# %%

f1_score(y_pred=df['Fraud_Heuristic'], y_true=df['isFraud'])

# %%

from sklearn.metrics import confusion_matrix

# %%

cm = confusion_matrix(y_pred=df['Fraud_Heuristic'], y_true=df['isFraud'])

# %%

plot_confusion_matrix(cm, ['Genuine', 'Fraud'], normalize=False)

# %%

df.shape

# %%
# 折合成24小时
df['hour'] = df['step'] % 24

# %%
# 按小时统计下欺诈和真实的数量
frauds = []
genuine = []
for i in range(24):
    f = len(df[(df['hour'] == i) & (df['isFraud'] == 1)])
    g = len(df[(df['hour'] == i) & (df['isFraud'] == 0)])
    frauds.append(f)
    genuine.append(g)

# %%


# %%

sns.set_style("white")

fig, ax = plt.subplots(figsize=(10, 6))
gen = ax.plot(genuine / np.sum(genuine), label='Genuine')
fr = ax.plot(frauds / np.sum(frauds), dashes=[5, 2], label='Fraud')
# frgen = ax.plot(np.devide(frauds,genuine),dashes=[1, 1], label='Fraud vs Genuine')
plt.xticks(np.arange(24))
legend = ax.legend(loc='upper center', shadow=True)
plt.show()
# fig.savefig('time.png')

# %%

sns.set_style("white")

fig, ax = plt.subplots(figsize=(10, 6))
# gen = ax.plot(genuine/np.sum(genuine), label='Genuine')
# fr = ax.plot(frauds/np.sum(frauds),dashes=[5, 2], label='Fraud')
frgen = ax.plot(np.divide(frauds, np.add(genuine, frauds)), label='Share of fraud')
plt.xticks(np.arange(24))
legend = ax.legend(loc='upper center', shadow=True)
plt.show()
# fig.savefig('time_comp.png')

# %%
# 检测诈骗者是否转账提款到他们自己的银行账户
dfFraudTransfer = df[(df.isFraud == 1) & (df.type == 'TRANSFER')]
dfFraudCashOut = df[(df.isFraud == 1) & (df.type == 'CASH_OUT')]
# 从结果看，没有
dfFraudTransfer.nameDest.isin(dfFraudCashOut.nameOrig).any()

# %%

dfNotFraud = df[(df.isFraud == 0)]
dfFraud = df[(df.isFraud == 1)]

dfFraudTransfer.loc[dfFraudTransfer.nameDest.isin(dfNotFraud.loc[dfNotFraud.type == 'CASH_OUT'].nameOrig.drop_duplicates())]

# %%

len(dfFraud[(dfFraud.oldBalanceDest == 0) & (dfFraud.newBalanceDest == 0) & (dfFraud.amount)]) / (1.0 * len(dfFraud))

# %%

len(dfNotFraud[(dfNotFraud.oldBalanceDest == 0) & (dfNotFraud.newBalanceDest == 0) & (dfNotFraud.amount)]) / (
            1.0 * len(dfNotFraud))

# %%

dfOdd = df[(df.oldBalanceDest == 0) &
           (df.newBalanceDest == 0) &
           (df.amount)]

# %%

len(dfOdd[(dfOdd.isFraud == 1)]) / len(dfOdd)

# %%
# 检测交易的源账户是否有足够的资金
len(dfOdd[(dfOdd.oldBalanceOrig <= dfOdd.amount)]) / len(dfOdd)

# %%

len(dfOdd[(dfOdd.oldBalanceOrig <= dfOdd.amount) & (dfOdd.isFraud == 1)]) / len(dfOdd[(dfOdd.isFraud == 1)])

# %%

dfOdd.columns

# %%

dfOdd.head(20)

# %%

df.head()

# %%
# type列全部增加前缀
df['type'] = 'type_' + df['type'].astype(str)

# %%

# Get dummies
dummies = pd.get_dummies(df['type'])

# Add dummies to df
df = pd.concat([df, dummies], axis=1)

# remove original column
del df['type']

# %% md
# Predictive modeling with Keras

# %%

df = df.drop(['nameOrig', 'nameDest', 'Fraud_Heuristic'], axis=1)

# %%

df['isNight'] = np.where((2 <= df['hour']) & (df['hour'] <= 6), 1, 0)

# %%

df[df['isNight'] == 1].isFraud.mean()

# %%

df.head()

# %%

df = df.drop(['step', 'hour'], axis=1)

# %%

df.head()

# %%

df.columns.values

# %%

y_df = df['isFraud']
x_df = df.drop('isFraud', axis=1)

# %%

y = y_df.values
X = x_df.values

# %%

y.shape

# %%

X.shape

# %%

from sklearn.model_selection import train_test_split

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

# %%

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.1,
                                                  random_state=42)

# %%

from imblearn.over_sampling import SMOTE, RandomOverSampler

# %%

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# %%

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

# %%

# Log reg
model = Sequential()
model.add(Dense(1, input_dim=9))
model.add(Activation('sigmoid'))

# %%

model.summary()

# %%

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-5),
              metrics=['acc'])

# %%

model.fit(X_train_res, y_train_res,
          epochs=5,
          batch_size=256,
          validation_data=(X_val, y_val))

# %%

y_pred = model.predict(X_test)

# %%

y_pred[y_pred > 0.5] = 1
y_pred[y_pred < 0.5] = 0

# %%

f1_score(y_pred=y_pred, y_true=y_test)

# %%

cm = confusion_matrix(y_pred=y_pred, y_true=y_test)

# %%

plot_confusion_matrix(cm, ['Genuine', 'Fraud'], normalize=False)

# %%

model = Sequential()
model.add(Dense(16, input_dim=9))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# %%

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-4), metrics=['acc'])

# %%

model.fit(X_train_res, y_train_res,
          epochs=5, batch_size=256,
          validation_data=(X_val, y_val))

# %%

y_pred = model.predict(X_test)

# %%

y_pred[y_pred > 0.5] = 1
y_pred[y_pred < 0.5] = 0

# %%

f1_score(y_pred=y_pred, y_true=y_test)

# %%

cm = confusion_matrix(y_pred=y_pred, y_true=y_test)

# %%

plot_confusion_matrix(cm, ['Genuine', 'Fraud'], normalize=False)

# %% md

# Tree based methods

# %%

from sklearn.tree import export_graphviz

# %%

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# %%

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as PImage

# import pydotplus
dot_data = StringIO()
'''export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)'''
with open("tree1.dot", 'w') as f:
    f = export_graphviz(dtree,
                        out_file=f,
                        max_depth=3,
                        impurity=True,
                        feature_names=list(df.drop(['isFraud'], axis=1)),
                        class_names=['Genuine', 'Fraud'],
                        rounded=True,
                        filled=True)

# Convert .dot to .png to allow display in web notebook
check_call(['dot', '-Tpng', 'tree1.dot', '-o', 'tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
img.save('sample-out.png')
PImage("sample-out.png")

# %%

from sklearn.ensemble import RandomForestClassifier

# %%

rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
rf.fit(X_train, y_train)

# %%

y_pred = rf.predict(X_test)

# %%

f1_score(y_pred=y_pred, y_true=y_test)

# %%

cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
plot_confusion_matrix(cm, ['Genuine', 'Fraud'], normalize=False)

# %%

import xgboost as xgb

# %%

booster = xgb.XGBClassifier(n_jobs=-1)
booster = booster.fit(X_train, y_train)

# %%

y_pred = booster.predict(X_test)

# %%

f1_score(y_pred=y_pred, y_true=y_test)

# %%

cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
plot_confusion_matrix(cm, ['Genuine', 'Fraud'], normalize=False)

# %% md

# Entity embeddings

# %%

# Reload data
df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
df = df.rename(columns={'oldbalanceOrg': 'oldBalanceOrig', 'newbalanceOrig': 'newBalanceOrig', \
                        'oldbalanceDest': 'oldBalanceDest', 'newbalanceDest': 'newBalanceDest'})

# %%

df.head()

# %%

df = df.drop(['nameDest', 'nameOrig', 'step'], axis=1)

# %%

df['type'].unique()

# %%

map_dict = {}
for token, value in enumerate(df['type'].unique()):
    map_dict[value] = token

# %%

map_dict

# %%

df["type"].replace(map_dict, inplace=True)

# %%

df.head()

# %%

other_cols = [c for c in df.columns if ((c != 'type') and (c != 'isFraud'))]

# %%

other_cols

# %%

from keras.models import Model
from keras.layers import Embedding, Merge, Dense, Activation, Reshape, Input, Concatenate

# %%

num_types = len(df['type'].unique())
type_embedding_dim = 3

# %%

inputs = []
outputs = []

# %%

type_in = Input(shape=(1,))
type_embedding = Embedding(num_types, type_embedding_dim, input_length=1)(type_in)
type_out = Reshape(target_shape=(type_embedding_dim,))(type_embedding)

type_model = Model(type_in, type_out)

inputs.append(type_in)
outputs.append(type_out)

# %%

num_rest = len(other_cols)

# %%

rest_in = Input(shape=(num_rest,))
rest_out = Dense(16)(rest_in)

rest_model = Model(rest_in, rest_out)

inputs.append(rest_in)
outputs.append(rest_out)

# %%

concatenated = Concatenate()(outputs)

# %%

x = Dense(16)(concatenated)
x = Activation('sigmoid')(x)
x = Dense(1)(concatenated)
model_out = Activation('sigmoid')(x)

# %%

merged_model = Model(inputs, model_out)
merged_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

# %%

types = df['type']

# %%

rest = df[other_cols]

# %%

target = df['isFraud']

# %%

history = merged_model.fit([types.values, rest.values], target.values,
                           epochs=1, batch_size=128)

# %%

merged_model.summary()

# %%




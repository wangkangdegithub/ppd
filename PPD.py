# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

data1=pd.read_csv('E:/kaggle数据/PPD/Training Set/PPD_LogInfo_3_1_Training_Set.csv')
data2=pd.read_csv('E:/kaggle数据/PPD/Training Set/PPD_Training_Master_GBK_3_1_Training_Set.csv',encoding='gb18030')
data3=pd.read_csv('E:/kaggle数据/PPD/Training Set/PPD_Userupdate_Info_3_1_Training_Set.csv')
'''
data1.head()
data2.head()
data3.head()
data2.isnull().sum().sort_values(ascending=False).head(100)

'''
data2=data2.drop(['WeblogInfo_1','WeblogInfo_3'],axis=1)
#查看data2 中UserInfo_11——UserInfo_13中数值类型，缺失值   以新类型 ‘-1’ 补全
print(data2.dtypes)
data2.UserInfo_11.value_counts()
data2.UserInfo_12.value_counts()
data2.UserInfo_13.value_counts()
data2.UserInfo_11=data2.UserInfo_11.fillna('-1.0')
data2.UserInfo_12=data2.UserInfo_12.fillna('-1.0')
data2.UserInfo_13=data2.UserInfo_13.fillna('-1.0')

#查看data2 中UserInfo_2/UserInfo_4缺失值用 ‘不详’ 补全
data2.UserInfo_4=data2.UserInfo_4.fillna('不详')
data2.UserInfo_2=data2.UserInfo_2.fillna('不详')

#查看data2 中UserInfo_1/UserInfo_3缺失值以 众数 补全
data2.UserInfo_1=data2.UserInfo_1.fillna(data2.UserInfo_1.mode()[0])
data2.UserInfo_3=data2.UserInfo_3.fillna(data2.UserInfo_1.mode()[0])

#观察data2 发现所有缺失值都是数值型，所以全以 众数 补全
data2 = data2.fillna(data2.mode().T[0])
#观察确认 data2中缺失值情况
data2.isnull().sum().sort_values(ascending=False)

#观察data2中违约人数的分布target,发现这是一个样本不平衡问题
data2.target.value_counts()
'''
#(待完善)
#统计data2 每一行的缺失值个数,按Idx,Null_Count排列
#x = []
y = []
for i in range(0,len(data2)):
    x.append(i)
    y.append(data2.loc[i,:].isnull().sum())
dx = pd.Series(x)
Null_Count = pd.Series(y)
cc=DataFrame({'Idx':Idx,'Null_count':Null_count})
'''
#data2 中空格符处理
data2['UserInfo_9']=data2['UserInfo_9'].str.strip()
#城市名处理,去掉'市'字
data2.UserInfo_8=data2.UserInfo_8.str.rstrip('市')
data2.UserInfo_8=data2.UserInfo_8.str.strip()
#省名处理,去掉'省'字，‘市’字
data2.UserInfo_19=data2.UserInfo_19.str.rstrip('省')
data2.UserInfo_19=data2.UserInfo_19.str.rstrip('市')
data2.UserInfo_19=data2.UserInfo_19.str.strip()

'''
# data2存到本地，用sql增删改查
data2.to_csv('E:/kaggle数据/PPD/Middle1_PPD_LogInfo_3_1_Training_Set.csv')
'''
#data3中字符大小写处理  大写转为小写
data3.UserupdateInfo1=data3.UserupdateInfo1.str.lower()
data3.UserupdateInfo1.value_counts()
# ↓↓↓↓↓↓↓特征工程处理↓↓↓↓↓↓↓
#观察违约率和省份之间关系，降低UserInfo_7/UserInfo_19的二值特征的维度,取违约率为前六的省市
dummies_UserInfo_7=pd.get_dummies(data2.UserInfo_7,prefix='UserInfo_7')
UserInfo_7_cols=['UserInfo_7_山东','UserInfo_7_天津','UserInfo_7_四川','UserInfo_7_湖南','UserInfo_7_海南','UserInfo_7_辽宁']
dummies_UserInfo_7=dummies_UserInfo_7.loc[:,UserInfo_7_cols]

dummies_UserInfo_19=pd.get_dummies(data2.UserInfo_19,prefix='UserInfo_19')
UserInfo_19_cols=['UserInfo_19_天津','UserInfo_19_山东','UserInfo_19_吉林','UserInfo_19_黑龙江','UserInfo_19_辽宁','UserInfo_19_湖南']
dummies_UserInfo_19=dummies_UserInfo_19.loc[:,UserInfo_19_cols]
# 把UserInfo_8的城市one-hot后，带来高维灾难，通过计算feature_importances，降低维数。
dummies_UserInfo_8=pd.get_dummies(data2.UserInfo_8,prefix='UserInfo_8')
X=dummies_UserInfo_8
y=data2.target

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

rfr=RandomForestClassifier()
rfr.fit(X,y)
importances = rfr.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfr.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
ans = indices[0:10]

#取 通过计算feature_importances为前10的特征（把UserInfo_8的城市） 并可视化
print("Feature ranking:")
for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# draw histogram of feature importances
'''
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[ans],color="r", yerr=std[ans], align="center")
plt.xticks(range(10), ans)
plt.xlim([-1, 10])
plt.show()
'''
#构建feature_importances dataframe
importances_df = DataFrame({'cols':X.columns,'imps':importances})
importances_df_top20 = importances_df.sort(columns='imps',ascending=False).head(20)

# 通过观察 取feature_importances在前20的城市作为新的二值特征向量
importances_top20_cols = []
for i in importances_df_top20.cols:
    importances_top20_cols.append(i)

dummies_UserInfo_8 = dummies_UserInfo_8.loc[:, importances_top20_cols]

# 对于UserInfo_2/4/20 这三个  城市信息字段，把其离散化为一线二线三线城市，同样需要One-hot编码。
# 首先，对UserInfo_20进行处理，把‘市’字去掉
data2.UserInfo_20 = data2.UserInfo_20.str.rstrip('市')
# 可以采取以下转换城市字段的方式：
# 1.按城市等级划分为1、2、3 线城市 （UserInfo_2）
# 2.城市特征向量化                （UserInfo_4）
# 3.地理位置差异特征              （UserInfo_20）
city_1 = ['北京', '上海', '广州', '深圳']
city_2 = ['杭州', '南京', '济南', '重庆', '青岛', '大连', '宁波', '厦门', '成都', '武汉', '哈尔滨', '沈阳', '西安', '长春', '长沙',
          '福州', '郑州', '石家庄', '苏州', '佛山', '东莞', '无锡', '烟台', '太原', '合肥', '南昌', '昆明', '温州', '淄博', '唐山']
city_1_2 = ['一线城市', '二线城市']


# step1、转换为一线、二线城市
def city_convert(UserInfo):
    for i in city_1:
        data2.loc[:, UserInfo].replace(i, '一线城市', inplace=True)
    for i in city_2:
        data2.loc[:, UserInfo].replace(i, '二线城市', inplace=True)


UserInfo_city = ['UserInfo_2', 'UserInfo_4', 'UserInfo_20']
for paraments in UserInfo_city:
    city_convert(paraments)
    print('%d 数据转换成功~')

List_UserInfo_2 = data2.UserInfo_2.unique().tolist()
List_UserInfo_4 = data2.UserInfo_4.unique().tolist()
List_UserInfo_20 = data2.UserInfo_20.unique().tolist()

# step2、转换为三线城市
UserInfo = ['UserInfo_2', 'UserInfo_4', 'UserInfo_20']


def city_convert_three(List_UserInfo):
    for i in List_UserInfo:
        if i not in city_1_2:
            for j in UserInfo:
                data2.loc[:, j].replace(i, '三线城市', inplace=True)


city_convert_three(List_UserInfo_2)
city_convert_three(List_UserInfo_4)
city_convert_three(List_UserInfo_20)
#分别观察UserInfo_2，UserInfo_4，UserInfo_20 修改后的分布
#发现UserInfo_20修改后全为三线城市：说明UserInfo_20不应该这样划分
data2.UserInfo_4.value_counts()
#待完善
dummies_UserInfo_2=pd.get_dummies(data2.UserInfo_2,prefix='UserInfo_2')
dummies_UserInfo_4=pd.get_dummies(data2.UserInfo_4,prefix='UserInfo_4')
dummies_UserInfo_20=pd.get_dummies(data2.UserInfo_20,prefix='UserInfo_20')
#由于此前按三线城市对UserInfo_20划分，而UserInfo_20字段中权威三线城市，所以此字段对结果无影响，可以drop()删去
#观察 ListingInfo（成交时间）与target(是否违约)之间的关系
import matplotlib.pyplot as plt
from datetime import date
train_1 = data2[data2.target==1]
temp = train_1[['ListingInfo'  ,'target']].groupby('ListingInfo').agg('sum')*2
temp = temp.rename(columns={'target':'count_1'})
temp['date'] = temp.index

temp.date = temp.date.apply(lambda x:(date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))-date(2013,11,1)).days)
temp = temp.sort(columns='date')

ax = temp.plot(x='date',y='count_1',title="train set")

train_0 = data2[data2.target==0]
#train_0.target = [1 for _ in range(len(train_0))]
temp_0 = train_0[['ListingInfo','target']].groupby('ListingInfo').agg('count')
temp_0 = temp_0.rename(columns={'target':'count_0'})
temp_0['date'] = temp_0.index
temp_0.date = temp_0.date.apply(lambda x:(date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))-date(2013,11,1)).days)
temp_0 = temp_0.sort(columns='date')

temp_0.plot(x='date',y='count_0',ax=ax)

'''
plt.xlabel('Date(20131101~20141109)')
plt.ylabel('count')
plt.show()
plt.close()
'''
#通过观察 发现违约率与成交时间有关，因此把ListingInfo日期离散化
# 测试****************
date = temp.date
# 成交日期变为步长为10的离散属性
bins = []
for i in range(0, 380, 10):
    bins.append(i)

group_names = []
for i in range(1, len(bins)):
    group_names.append(i)
discrete_temp_date = pd.cut(date, bins, right=False, labels=group_names)
print(discrete_temp_date)

# 新增一个date列，存放交易时间的离散值
from datetime import date

data2['date'] = data2.ListingInfo.apply(
    lambda x: (date(int(x.split('/')[0]), int(x.split('/')[1]), int(x.split('/')[2])) - date(2013, 11, 1)).days)
new_date = data2['date']

# 成交日期变为步长为10的离散属性
bins = []
for i in range(0, 390, 10):
    bins.append(i)

group_names = []
for i in range(1, len(bins)):
    group_names.append(i)

discrete_temp_date = pd.cut(new_date, bins, right=False, labels=group_names)
data2.date = discrete_temp_date

#data2中的ListingInfo字段被date字段所替换
data2.drop('ListingInfo',axis=1,inplace=True)

new_data2=pd.concat([data2,dummies_UserInfo_7,dummies_UserInfo_19,dummies_UserInfo_8,dummies_UserInfo_2,dummies_UserInfo_4],axis=1)
#通过观察发现UserInfo_24的特征取值95%都是‘D’，所以可以删除这一字段drop()
data2.UserInfo_24.value_counts()
new_data2=new_data2.drop(['UserInfo_7','UserInfo_19','UserInfo_8','UserInfo_2','UserInfo_4','UserInfo_20','UserInfo_24'],axis=1)

'''
#data2中需要添加的字段
dummies_UserInfo_7
dummies_UserInfo_19
dummies_UserInfo_8
dummies_UserInfo_2
dummies_UserInfo_4
date

#需要删除
UserInfo_7
UserInfo_19
UserInfo_8
UserInfo_2
UserInfo_4
UserInfo_20
ListingInfo
'''
#step1: 确认变量类型 ,object or int/float?
#step2: 确认int/float 取值范围，是否需要转换int/float为object 以便 one-hot
#data2_cols = data2.columns
new_data2_object_cols = new_data2.columns[new_data2.dtypes == 'object']
new_data2_int_cols = new_data2.columns[new_data2.dtypes != 'object']
dummies_data2_object=pd.get_dummies(new_data2[new_data2_object_cols],prefix=new_data2_object_cols)
new_data2=pd.concat([new_data2,dummies_data2_object],axis=1)
#修改后的data2数据集
new_data2=new_data2.drop(new_data2_object_cols,axis=1)
new_data2.to_csv('E:/kaggle数据/PPD/new_data2.csv',index=False)
print('new_data2保存到本地')

#data2数据格式整理结束 需要把data1/data2/data3通过间Idx连接起来
#首先观察data1/data3的特征，由于之前已光测过data1/data3都没有缺失值，因此需要看看是否需要one-hot
data1_objects_columns=data1.columns[data1.dtypes =='object']
data3_objects_columns=data3.columns[data3.dtypes =='object']

#确认以下三个维度：每个Idx 1、登录次数 LogInfo3  2、每种操作代码的次数 LogInfo1  3、操作类别的次数 LogInfo2
import collections
train_loginfo = pd.read_csv('E:/kaggle数据/PPD/Training Set/PPD_LogInfo_3_1_Training_Set.csv')
Loginfo_number = collections.defaultdict(list) ### 用户操作的次数
Loginfo_category = collections.defaultdict(set) ###用户操作的种类数
Loginfo_times = collections.defaultdict(list) ### 用户分登录次数
Loginfo_date = collections.defaultdict(list) #### 用户借款成交与登录时间跨度

with open('E:/kaggle数据/PPD/Training Set/PPD_LogInfo_3_1_Training_Set.csv' ,'r') as f:
    f.readline().strip().split(",")
    for line in f:
        cols = line.strip().split(",") ### cols 是list结果
        Loginfo_date[cols[0]].append(cols[1])
        Loginfo_number[cols[0]].append(cols[2])
        Loginfo_category[cols[0]].add(cols[3])
        Loginfo_times[cols[0]].append(cols[4])
    print(u'提取信息完成')
import datetime as dt
Loginfo_number_ = collections.defaultdict(int)  ### 用户操作的次数
Loginfo_category_ = collections.defaultdict(int) ###用户操作的种类数
Loginfo_times_ = collections.defaultdict(int)  ### 用户分登录次数
Loginfo_date_ = collections.defaultdict(int) #### 用户借款成交与登录时间跨度

for key in Loginfo_date.keys():
    Loginfo_times_[key] = len(Loginfo_times[key])
    Loginfo_delta_date = dt.datetime.strptime(Loginfo_date[key][0] ,'%Y-%m-%d') - dt.datetime.strptime(list(set(Loginfo_times[key]))[0] ,'%Y-%m-%d')
    #if delta_date.days  >=0 :
    Loginfo_date_[key] = abs(Loginfo_delta_date.days)  #abs() 函数返回数字的绝对值。
    #else:
        #loginfo_delta_date_ = dt.datetime.strptime(loginfo_date[key][0] ,'%Y/%m/%d') - dt.datetime.strptime(list(set(loginfo_times[key]))[-1] ,'%Y/%m/%d')
        #loginfo_date_[key] = abs(delta_date_.days)
    Loginfo_number_[key] = len(Loginfo_number[key])
    Loginfo_category_[key] = len(Loginfo_category[key])

print('信息处理完成')

## 建立一个DataFrame  data1转为需要的格式 Loginfo
Log_Idx_ = Loginfo_date_.keys() #### list
Log_numbers_ = Loginfo_number_.values()
Log_categorys_ = Loginfo_category_.values()
Log_times_ = Loginfo_times_.values()
Log_dates_ = Loginfo_date_.values()
Loginfo_df = pd.DataFrame({'Idx':list(Log_Idx_) , 'Log_numbers':list(Log_numbers_) ,'Log_categorys':list(Log_categorys_ ),'Log_times':list(Log_times_ ),'Log_dates':list(Log_dates_) })

#data1表变为Loginfo_df表
Loginfo_df.head()
Loginfo_df.to_csv('E:/kaggle数据/PPD/new_data1.csv',index=False,encoding='utf-8')
print('new_data1保存到本地')

##  userupdateinfo表
Userupdate_info_number = collections.defaultdict(list)  ### 用户信息更新的次数
Userupdate_info_category = collections.defaultdict(set)  ###用户信息更新的种类数
Userupdate_info_times = collections.defaultdict(list)  ### 用户分几次更新了
Userupdate_info_date = collections.defaultdict(list)  #### 用户借款成交与信息更新时间跨度

with open('E:/kaggle数据/PPD/Training Set/PPD_Userupdate_Info_3_1_Training_Set.csv', 'r') as f:
    f.readline().strip().split(",")
    for line in f:
        cols = line.strip().split(",")  ### cols 是list结果
        Userupdate_info_date[cols[0]].append(cols[1])
        Userupdate_info_number[cols[0]].append(cols[2])
        Userupdate_info_category[cols[0]].add(cols[2])
        Userupdate_info_times[cols[0]].append(cols[3])
    print(u'提取信息完成')

Userupdate_info_number_ = collections.defaultdict(int)  ### 用户信息更新的次数
Userupdate_info_category_ = collections.defaultdict(int)  ###用户信息更新的种类数
Userupdate_info_times_ = collections.defaultdict(int)  ### 用户分几次更新了
Userupdate_info_date_ = collections.defaultdict(int)  #### 用户借款成交与信息更新时间跨度

for key in Userupdate_info_date.keys():
    # 注意 set()的用法，对于列List中相同的元素，取唯一值（去重后的值）
    # 用户更新几次，取决于‘修改时间’UserupdateInfo2有几个不同的值，可用set（）函数解决
    Userupdate_info_times_[key] = len(set(Userupdate_info_times[key]))

    # 注意 set()的用法，对于列List中相同的元素，取唯一值（去重后的值）
    delta_date = dt.datetime.strptime(Userupdate_info_date[key][0], '%Y/%m/%d') - dt.datetime.strptime(
        list(set(Userupdate_info_times[key]))[0], '%Y/%m/%d')
    # if delta_date.days  >=0 :
    Userupdate_info_date_[key] = abs(delta_date.days)
    # else:
    # delta_date_ = dt.datetime.strptime(userupdate_info_date[key][0] ,'%Y/%m/%d') - dt.datetime.strptime(list(set(userupdate_info_times[key]))[0] ,'%Y/%m/%d')
    # userupdate_info_date_[key] = abs(delta_date_.days)

    Userupdate_info_number_[key] = len(Userupdate_info_number[key])
    Userupdate_info_category_[key] = len(Userupdate_info_category[key])

print('信息处理完成')

## 建立一个DataFrame
Idx_ = Userupdate_info_date_.keys()  #### list
numbers_ = Userupdate_info_number_.values()
categorys_ = Userupdate_info_category_.values()
times_ = Userupdate_info_times_.values()
dates_ = Userupdate_info_date_.values()
Userupdate_df = pd.DataFrame(
    {'Idx': list(Idx_), 'numbers': list(numbers_), 'categorys': list(categorys_), 'times': list(times_),
     'dates': list(dates_)})

# data3表变为Userupdate_df表
Userupdate_df.head()
Userupdate_df.to_csv('E:/kaggle数据/PPD/new_data3.csv', index=False, encoding='utf-8')
print('new_data3保存到本地')
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓合并三个表 ↓↓↓↓↓↓↓↓↓↓
a1=pd.read_csv('E:/kaggle数据/PPD/new_data1.csv')
a2=pd.read_csv('E:/kaggle数据/PPD/new_data2.csv',encoding='gb18030')
a3=pd.read_csv('E:/kaggle数据/PPD/new_data3.csv')
all_data=pd.merge(a1,a2,how='left',on='Idx')
all_data=pd.merge(a3,all_data,how='left',on='Idx')
all_data=all_data.fillna(all_data.median())
#处理训练样本分布不均衡问题
from sklearn.cross_validation import train_test_split

believe_indices = all_data[all_data.target == 0.0].index
random_indices = np.random.choice(believe_indices, 2117, replace=False)
believe_sample = all_data.loc[random_indices]
unbelieve_indices = all_data[all_data.target == 1.0].index
unbelieve_sample = all_data.loc[unbelieve_indices]
new_all_data=pd.concat([believe_sample,unbelieve_sample],axis=0)

y=new_all_data.target
X=new_all_data.drop('target',axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
rfr=RandomForestClassifier(n_estimators =100,max_features=20,max_depth=None)
model=rfr.fit(X_train,y_train)
from sklearn.cross_validation import cross_val_score
score=cross_val_score(model,X_train,y_train,cv=5)
result=model.predict(X_test)
print(score)
from sklearn.externals import joblib
joblib.dump(model, "E:/kaggle数据/PPD/model/train_model.m")
print('模型保存到本地')

'''
#根据学习曲线查看拟合状态
from sklearn.learning_curve import learning_curve
# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1,train_sizes=np.linspace(.05, 1., 30), verbose=0, plot=True):

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("samples")
        plt.ylabel('score')
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="train_set_score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="cross_test_score")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff
train_sizes, train_scores, test_scores=learning_curve(model,X_train,y_train,cv=5)
plot_learning_curve(model, "learning curve", X_train,y_train)
'''

'''
#利用网格搜索法寻找最佳参数
from sklearn.grid_search import GridSearchCV
tuned_parameters = [{'n_estimators':[10,100,500], 'max_features':[20,30,40] }]

scores = ['r2']
for score in scores:
    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)
    print("正在计算最佳参数：")
    #best_estimator_ returns the best estimator chosen by the search
    print(clf.best_estimator_)
    print ("")
    print("得分分别是:")
    print ("")
    #grid_scores_的返回值:
    #    * a dict of parameter settings
    #    * the mean score over the cross-validation folds
    #    * the list of scores for each fold
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

'''

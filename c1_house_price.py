# -*- coding: utf-8 -*-
__author__ = 'junjun'

import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()



def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path,encoding="utf-8")


if not os.path.exists(HOUSING_PATH):
    fetch_housing_data()

housing = load_housing_data()
#print(housing)
housing.info()

#看看数据分布

# housing.hist(bins=50, figsize=(20,15))
# plt.show()

#---start---
#获取测试集合 (随机的办法)
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print("训练集"+str(len(train_set))+"个,测试集合"+str(len(test_set))+"个")
#---end---



#---start---
#由于专家说median income非常重要，在训练的时候需要把每一个median income level的人都要有，上面的随机分配可能出现问题。
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5) #将media_income分为5类
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True) #大于5的也当成5

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)

#根据income_cat来sample
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print ("总分布:\r\n",housing["income_cat"].value_counts() / len(housing))
print ("训练集分布:\r\n",strat_train_set["income_cat"].value_counts() / len(housing))

#用不上 income_cat了删除
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
#---end---


#---start---
#Discover and Visualize the Data to Gain Insights

#使用可视化，看看一pandas些模式
housing = strat_train_set.copy()
#按照经纬度看看房屋的数量
# housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)
#
# #看看房屋价格和人口
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#          s=housing["population"]/100, label="population",
#          c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#      )
# plt.show();
#---end---



#---start---
#standard correlation coecient 看看相关系数，各个系数之间的关系
corr_matrix = housing.corr()
#房屋价格和其他属性的相关性
#相关系数是从-1到1，0为不相关，1位正相关，-1位负相关
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#增加几个参数
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
#bedrooms_per_room          -0.256332 相关性还比较高
#---end---



#---start---
# 开始准备数据了
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#total_bedrooms 有些数据是空的,需要处理
#housing.dropna(subset=["total_bedrooms"])
#housing.drop("total_bedrooms", axis=1)
#median = housing["total_bedrooms"].median()
#housing["total_bedrooms"].fillna(median)

#处理非文本数据
from sklearn.preprocessing import Imputer
#ocean_proximity是文本类型删除，先删除
housing_num = housing.drop("ocean_proximity", axis=1)
imputer = Imputer(strategy="median")
imputer.fit(housing_num)

#各个属性的中位数
print("中位数:",imputer.statistics_)

#将各个属性的空值设置为中位数
X = imputer.transform(housing_num)

#X是plain Numpy array 需要转化为Pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


#开始处理文本和类别属性(非数字)
# Handling Text and Categorical Attributes

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]

housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded
#每一个文本代表的数字
print(encoder.classes_)

#由于讲文本编码为0，1，2，3，4这种类型的,会有问题，因为1和2比较近，但是实际上1和4也许更近，
#所以使用 [1,0,0,0,0]表示0， [0,0,0,0,1]表示4
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1hot.toarray())


#上面2个可以使用一个过程来表示
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot
print(encoder.classes_)
print(housing_cat_1hot)



#自定义Transformers
#rooms_per_household，population_per_household，bedrooms_per_room，三个属性，前面看到这个相关性比较大
#add_bedrooms_per_room 是一个hyperparameter,This hyperpara‐ meter will allow you to easily find out whether adding this attribute helps the Machine Learning algorithms or not.

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
print(housing_extra_attribs)

#---end---




#---start---
#Feature Scaling 归一化
#min-max scaling and standardization.

#---end---




#---start---
#Transformation Pipelines 转换管道
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")), #用中位值补足空数据
        ('attribs_adder', CombinedAttributesAdder()), #添加其他属性
        ('std_scaler', StandardScaler()), #归一化
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
print("#111\r\n",housing_num_tr)


#自顶一个Transformer来处理文本
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


#解决一个bug “TypeError: fit_transform() takes 2 positional arguments but 3 were given”
from sklearn.base import BaseEstimator, TransformerMixin
class LabelBinarizer_new(TransformerMixin, BaseEstimator):
    def fit(self, X, y = 0):
        return self
    def transform(self, X, y = 0):
        encoder = LabelBinarizer();
        result = encoder.fit_transform(X)
        return result;

#文本值和数值属性同时进行
from sklearn.pipeline import FeatureUnion
num_attribs = list(housing_num) #数值属性
cat_attribs = ["ocean_proximity"] #类别属性

num_pipeline = Pipeline([
     ('selector', DataFrameSelector(num_attribs)),
     ('imputer', Imputer(strategy="median")),
     ('attribs_adder', CombinedAttributesAdder()),
     ('std_scaler', StandardScaler()),])

cat_pipeline = Pipeline([
     ('selector', DataFrameSelector(cat_attribs)),
     ('label_binarizer', LabelBinarizer_new()),
    ])

full_pipeline = FeatureUnion(transformer_list=[
     ("num_pipeline", num_pipeline),
     ("cat_pipeline", cat_pipeline),])

#数据处理好了~
housing_prepared = full_pipeline.fit_transform(housing)

print("处理好的数据\r\n",housing_prepared)
#---end---





#---训练模型---

#---start---

#线性模型
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#现在得到了一个模型
#取一些数据出来看看效果
some_data = housing.iloc[:5000]
some_labels = housing_labels.iloc[:5000]

#数据清理
some_data_prepared = full_pipeline.transform(some_data)
print("some_data_prepared\r\n",some_data_prepared)

#模型输出的数据
housing_predictions = lin_reg.predict(some_data_prepared)
print("预测的数据",list(housing_predictions))
print("真实的数据",list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("线性模型的rmse:（均方差）",lin_rmse)


#使用随机森林
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
housing_predictions = forest_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("随机森林的rmse:（均方差）",lin_rmse)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)
#---end---



#---start
#Fine-Tune Your Model调试模型
#1.Grid Search
#规定参数，遍历所有参数组合，找到最好的
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]

#param_grid
'''
    param_grid第一行有3x4=12种组合，第二行有2x3种组合，所以会有12+6中组合
'''
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

#找到最好的参数组合
print(grid_search.best_params_)

#直接得到最好的模型
print(grid_search.best_estimator_)

#看看每一种组合的情况
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#---end


#---start
#Analyze the Best Models and Their Errors 分析模型
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
#列出每个属性的重要程度
#有些不重要的属性就可以删除了~
attr_importance = sorted(zip(feature_importances, attributes), reverse=True)
print(attr_importance)

#---end
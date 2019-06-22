#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'weill'
@date = '2019/06/22 0001'

# https://github.com/MLEveryday/100-Days-Of-ML-Code
"""
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

dataset = pd.read_csv('../dataset/data.csv', index_col=False)  # 读取数据
X = dataset.iloc[ : , :-1].values  # X: 除最后一列数据的所有数据（各属性值）
Y = dataset.iloc[ : , 3].values  # Y: 最后一列数据（类别)

# lijiancheng0614.github.io/scikit-learn/modules/generated/sklearn.preprocessing.Imputer.html
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)  # 实例化Imputer类，将'NaN'替换为(列)均值
imputer = imputer.fit(X[ : , 1:3])  # imputer实例用fit方法，处理X第2-3列中的丢失数据
X[ : , 1:3] = imputer.transform(X[ : , 1:3])  # 将转换处理后的数据重新赋值给X

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
labelencoder_X = LabelEncoder()  # 实例化LabelEncoder类，将数据标签化，利于模型的建立
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])  # 将X的第一列数据标签化(3种): [0, 1，2]
onehotencoder = OneHotEncoder(categorical_features = [0])  # 创建虚拟变量，将有n种标签的一个属性变成n个二元的特征
X = onehotencoder.fit_transform(X).toarray()  # 将X标签化数据变成: [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
labelencoder_Y = LabelEncoder()  # 实例化LabelEncoder类，将数据标签化
Y =  labelencoder_Y.fit_transform(Y)  # 将Y的数据标签化(2种): [0, 1]

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0)  # 将数据拆分为训练集和测试集

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
sc_X = StandardScaler()  # 实例化标准化StandarScalar类， zscore标准化：z = (x - u) / s （u: 均值，s： 方差）
X_train = sc_X.fit_transform(X_train)  # 对X训练集数据，进行拟合(计算均值和方差）后再进行标准化转换
X_test = sc_X.transform(X_test)  # 对X测试集进行标准化转换

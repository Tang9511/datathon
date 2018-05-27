import pandas as pd
from SBBTree_ONLINE import SBBTree
from sympy import integrate
import numpy as np
import math
import preprocessing

features = pd.read_csv('data/features(v2.3).csv')


major_dict = {
    'cs': 0,
    'business': 1,
    'nursing': 2,
    'science': 0,
    'engineering': 0,
    'bba': 1
}

university_dict = {
    'hku': 1,
    'cuhk': 2,
    'polyu': 3,
    'bu': 4
}

regression_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2'},
    'num_leaves': 7,
    'learning_rate': 0.3,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'min_data': 1,
    'min_data_in_bin': 1
}

def gaussian(x, u, sig):
    return np.exp(-(x - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig)


def get_lowerq(pred_lowerq, test_average):
    real_lowerq = []
    pred_lowerq = pred_lowerq.reshape((1, -1))
    test_average = test_average.reshape((1, -1))
    for i in range(test_average.shape[1]):
        if test_average[0, i] >= round(pred_lowerq[0, i]):
            real_lowerq.append(round(pred_lowerq[0, i]))
        elif test_average[0, i] >= pred_lowerq[0, i]:
            real_lowerq.append(int(pred_lowerq[0, i]))
        else:
            real_lowerq.append(int(test_average[0, i]))
    return np.array(real_lowerq).reshape((-1, 1))


def get_upperq(pred_upperq, test_average):
    real_upperq = []
    pred_upperq = pred_upperq.reshape((1, -1))
    test_average = test_average.reshape((1, -1))
    for i in range(test_average.shape[1]):
        if test_average[0, i] <= (int(pred_upperq[0, i])):
            print(test_average[0, i], int(pred_upperq[0, i]))
            real_upperq.append(int(pred_upperq[0, i]))
        else:
            real_upperq.append(math.ceil(test_average[0, i]))

    return np.array(real_upperq).reshape((-1, 1))


for key, value in major_dict.items():
    features['subject'][features['subject']==key] = value

def train_lowerq(features):
    train_data = features[features['university']!='polyu']
    test_data = features[features['university']=='polyu']

    for key, value in university_dict.items():
        train_data['university'][train_data['university']==key] = value
        test_data['university'][test_data['university'] == key] = value

    lowerq = np.array(list(train_data['lowerq'].values))

    test_average = np.array(list(test_data['average'].values))


    train_data = train_data.drop(columns=['lowerq', 'upperq'])
    test_data = test_data.drop(columns=['lowerq', 'upperq'])
    train_data = train_data.get_values()
    test_data = test_data.get_values()


    regression_model = SBBTree(params=regression_params,
                               stacking_num=1,
                               bagging_num=5,
                               bagging_test_size=0.2,
                               num_boost_round=10000,
                               early_stopping_rounds=200)


    regression_model.fit(train_data, lowerq)
    pred_lowerq = regression_model.predict(test_data)
    return get_lowerq(pred_lowerq, test_average), lowerq


def train_upperq(features):
    train_data = features[features['upperq'].notnull()]
    test_data = features[features['upperq'].isnull()]

    for key, value in university_dict.items():
        train_data['university'][train_data['university'] == key] = value
        test_data['university'][test_data['university'] == key] = value


    upperq = np.array(list(train_data['upperq'].values))

    test_average = np.array(list(test_data['average'].values))

    train_data = train_data.drop(columns=['lowerq', 'upperq'])
    test_data = test_data.drop(columns=['lowerq', 'upperq'])
    train_data = train_data.get_values()
    test_data = test_data.get_values()

    regression_model = SBBTree(params=regression_params,
                               stacking_num=1,
                               bagging_num=5,
                               bagging_test_size=0.2,
                               num_boost_round=10000,
                               early_stopping_rounds=200)

    regression_model.fit(train_data, upperq)
    pred_upperq = regression_model.predict(test_data)
    return get_upperq(pred_upperq, test_average), upperq


trained_upperq, atu_upperq = train_upperq(features)
trained_lowerq, atu_lowerq = train_lowerq(features)

trained_upperq = list(trained_upperq.reshape((1, -1))[0]) + list(atu_upperq)
trained_lowerq = list(trained_lowerq.reshape((1, -1))[0]) + list(atu_lowerq)

features1 = pd.read_csv('data/features(v2.3).csv')
features1['upperq'] = trained_upperq
features1['lowerq'] = trained_lowerq


def add_var_mean(features):
    lowerq = features['lowerq']
    average = features['average']
    real_average = features['real_average']
    var_dict, mean_dict = preprocessing.get_mean_var_dict()
    for i in range(0, len(features)):
        university = features.iloc[i]['university']
        subject = features.iloc[i]['subject']
        year = features.iloc[i]['year']
        key=university+'_'+subject+'_'+str(year)
        features.loc[(features['university'] == university)&(features['subject'] == subject)&(
            features['year'] == year), 'atu_mean'] = mean_dict[key]
        features.loc[(features['university'] == university) & (features['subject'] == subject) & (
        features['year'] == year), 'atu_var'] = var_dict[key]
    return features
features1 = add_var_mean(features1)



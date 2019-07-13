import pandas as pd
import numpy as np
import lightgbm as lgb

pd.set_option('display.max_columns',None)

#读取数据
data_path = 'data/'
age_train = pd.read_csv(data_path + "age_train.csv", names=['uid','age_group'])
age_test = pd.read_csv(data_path + "age_test.csv", names=['uid'])
user_basic_info = pd.read_csv(data_path + "user_basic_info.csv", names=['uid','gender','city','prodName','ramCapacity','ramLeftRation','romCapacity','romLeftRation','color','fontSize','ct','carrier','os'])
user_behavior_info = pd.read_csv(data_path + "user_behavior_info.csv", names=['uid','bootTimes','AFuncTimes','BFuncTimes','CFuncTimes','DFuncTimes','EFuncTimes','FFuncTimes','FFuncSum'])
user_app_actived = pd.read_csv(data_path + "user_app_actived.csv", names=['uid','appId'])
#user_app_usage = pd.read_csv("user_app_usage.csv")
app_info = pd.read_csv(data_path + "app_info.csv", names=['appId', 'category'])

#处理数据量较大的user_app_usage.csv，结合app_info.csv简单统计得到appuseProcessed.csv作为特征
def f(x):
    s = x.value_counts()
    return np.nan if len(s) == 0 else s.index[0]
def processUserAppUsage():
    resTable = pd.DataFrame()
    reader = pd.read_csv(data_path + "user_app_usage.csv", names=['uid','appId','duration','times','use_date'], iterator=True)
    last_df = pd.DataFrame()
    
    app_info = pd.read_csv(data_path + "app_info.csv", names=['appId','category'])
    cats = list(set(app_info['category']))
    category2id = dict(zip(sorted(cats), range(0,len(cats))))
    id2category = dict(zip(range(0,len(cats)), sorted(cats)))
    app_info['category'] = app_info['category'].apply(lambda x: category2id[x])
    i = 1
    
    while True:
        try:
            print("index: {}".format(i))
            i+=1
            df = reader.get_chunk(1000000)
            df = pd.concat([last_df, df])
            idx = df.shape[0]-1
            last_user = df.iat[idx,0]
            while(df.iat[idx,0]==last_user):
                idx-=1
            last_df = df[idx+1:]
            df = df[:idx+1]

            now_df = pd.DataFrame()
            now_df['uid'] = df['uid'].unique()
            now_df = now_df.merge(df.groupby('uid')['appId'].count().to_frame(), how='left', on='uid')
            now_df = now_df.merge(df.groupby('uid')['appId','use_date'].agg(['nunique']), how='left', on='uid')
            now_df = now_df.merge(df.groupby('uid')['duration','times'].agg(['mean','max','std']), how='left', on='uid')    

            now_df.columns = ['uid','usage_cnt','usage_appid_cnt','usage_date_cnt','duration_mean','duration_max','duration_std','times_mean','times_max','times_std']


            df = df.merge(app_info, how='left', on='appId')
            now_df = now_df.merge(df.groupby('uid')['category'].nunique().to_frame(), how='left', on='uid')
            #print(df.groupby(['uid'])['category'].value_counts().index[0])
            now_df['usage_most_used_category'] = df.groupby(['uid'])['category'].transform(f)
            resTable = pd.concat([resTable, now_df])
        except StopIteration:
            break
    
    resTable.to_csv("appuseProcessed.csv",index=0)
    
    print("Iterator is stopped")

processUserAppUsage()

    #将user_basic_info.csv 和 user_behavior_info.csv中的字符值编码成可以训练的数值类型，合并
class2id = {}
id2class = {}
def mergeBasicTables(baseTable):
    resTable = baseTable.merge(user_basic_info, how='left', on='uid', suffixes=('_base0', '_ubaf'))
    resTable = resTable.merge(user_behavior_info, how='left', on='uid', suffixes=('_base1', '_ubef'))
    cat_columns = ['city','prodName','color','carrier','os','ct']
    for c in cat_columns:
        resTable[c] = resTable[c].apply(lambda x: x if type(x)==str else str(x))
        sort_temp = sorted(list(set(resTable[c])))  
        class2id[c+'2id'] = dict(zip(sort_temp, range(1, len(sort_temp)+1)))
        id2class['id2'+c] = dict(zip(range(1,len(sort_temp)+1), sort_temp))
        resTable[c] = resTable[c].apply(lambda x: class2id[c+'2id'][x])
        
    return resTable

def get_user_app_actived_count_df():
    user_app_actived = pd.read_csv(data_path + "user_app_actived.csv", names=['uid','appId'])
    app_info = pd.read_csv(data_path + "app_info.csv", names=['appId', 'category'])

    # 得到 user_app_actived_list 二维数据，user_app_actived_list[0] 代表第0行的数据，
    # user_app_actived_list[0][0] 代表第0行的用户id，user_app_actived_list[0][1] 代表第0行的所有app id
    user_app_actived_list = []
    user_app_actived_list = user_app_actived.values
    del user_app_actived

    app_info_list = []
    app_info_list = app_info.values
    del app_info

    # 把应用分类信息存在字典里，键为'a006'之类的app id，值为一个list，里面存着一个app对应的多个类别
    app_category_dict = {}
    # 用字典存储出现过的所有类别，键为'社交通讯'之类的类别，值为序号，从0开始
    all_category_dict = {}
    all_category_name = []
    # all_category_len值最终为40，即有40类
    all_category_len = 0
    for temp_record in app_info_list:
        # 处理 app_category_dict
        if app_category_dict.get(temp_record[0], None) == None:
            category = []
            category.append(temp_record[1])
            app_category_dict[temp_record[0]] = category
        else:
            app_category_dict[temp_record[0]].append(temp_record[1])

        # 处理 all_category_dict
        if all_category_dict.get(temp_record[1], None) == None:
            all_category_dict[temp_record[1]] = all_category_len
            all_category_len = all_category_len + 1
            all_category_name.append(temp_record[1])
    del app_info_list
    
    # 统计每个用户安装的每一类别app的数量，以每一类别的数量作为特征
    # 第一列是用户id，后面第二列到41列是40个类别的app数量
    user_app_actived_count = np.zeros((user_app_actived_list.__len__(), 41))
    user_app_actived_count_len = 0
    # 遍历每一行数据
    for temp_record in user_app_actived_list:
        user_app_actived_count[user_app_actived_count_len][0] = temp_record[0]
        temp_app_list = temp_record[1].split('#')
        for app_id in temp_app_list:
            if app_category_dict.get(app_id, None) == None:
                continue
            else:
                temp_app_category_list = app_category_dict[app_id]
                for temp_app_category in temp_app_category_list:
                    category_index = all_category_dict[temp_app_category]
                    user_app_actived_count[user_app_actived_count_len][category_index + 1] += 1

        user_app_actived_count_len = user_app_actived_count_len + 1
    del user_app_actived_list

    column_name = ['uid']
    for temp_name in all_category_name:
        column_name.append(temp_name)
    user_app_actived_count_df = pd.DataFrame(user_app_actived_count, columns=column_name)
    return user_app_actived_count_df

#处理app使用相关数据
#对user_app_actived.csv简单统计
#将之前训练的appuseProcess.csv进行合并
def mergeAppData(baseTable):
    user_app_actived_count_df = get_user_app_actived_count_df()
    resTable = baseTable.merge(user_app_actived_count_df, how='left', on='uid')
    appusedTable = pd.read_csv("appuseProcessed.csv")
    resTable = resTable.merge(appusedTable, how='left', on='uid')
    resTable[['category', 'usage_most_used_category']] = resTable[['category', 'usage_most_used_category']].fillna(41)
    resTable = resTable.fillna(0)
    #print(resTable[:5])
    return resTable

#合并用户基本特征以及app使用相关特征，作为训练集和测试集
df_train = mergeAppData(mergeBasicTables(age_train))
df_test = mergeAppData(mergeBasicTables(age_test))
print(df_train.shape)
print(df_test.shape)
print(df_train.iloc[0:2,:])
print(df_test.iloc[0:2,:])


#训练模型

from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

print("训练模型：")
param = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 20,
        'objective': 'multiclass',
        'num_class': 7,
        'num_leaves': 31,
        'min_data_in_leaf': 50,
        'max_bin': 230,
        'feature_fraction': 0.8,
        'metric': 'multi_error'
        }

X = df_train.drop(['age_group','uid'], axis=1)
y = df_train['age_group']
uid = df_test['uid']
test = df_test.drop('uid', axis=1)

xx_score = []
cv_pred = []
skf = StratifiedKFold(n_splits=3, random_state=1030, shuffle=True)
for index, (train_index, vali_index) in enumerate(skf.split(X, y)):
    print(index)
    x_train, y_train, x_vali, y_vali = np.array(X)[train_index], np.array(y)[train_index], np.array(X)[vali_index], np.array(y)[vali_index]
    train = lgb.Dataset(x_train, y_train)
    vali =lgb.Dataset(x_vali, y_vali)
    print("training start...")
    model = lgb.train(param, train, num_boost_round=1000, valid_sets=[vali], early_stopping_rounds=50)
    xx_pred = model.predict(x_vali,num_iteration=model.best_iteration)
    xx_pred = [np.argmax(x) for x in xx_pred]
    xx_score.append(f1_score(y_vali,xx_pred,average='weighted'))
    y_test = model.predict(test,num_iteration=model.best_iteration)
    y_test = [np.argmax(x) for x in y_test]
    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))
        
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))
df = pd.DataFrame({'id':uid.as_matrix(),'label':submit})
df.to_csv('submission.csv',index=False)

age_train['age_group'].nunique()

# coding=utf-8
#!/usr/bin/python3
'''
Created on 2019年12月19日

@author: liushouhua
'''
import os
import gc
import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from scipy.stats import entropy
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_colwidth",100)
pd.set_option('display.width',1000)

cate_features = [ 'pos', 'app_version', 'device_vendor', 'netmodel', 'osversion', "hour"]

def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df

def leak_feats():
    print('=============================================== read train ===============================================')
    t = time.time()
    train_df = pd.read_pickle("data/train.pickle")
    train_df['date'] = pd.to_datetime(
        train_df['ts'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
    )
    train_df["split"] = train_df["ts"].apply(lambda x: datetime.utcfromtimestamp(x//1000).strftime("%d%H"))
    
    train_df.loc[train_df["split"]<="0815", 'day'] = 8
    train_df.loc[(train_df["split"]>"0815")&(train_df["split"]<="0915"), 'day'] = 9
    train_df.loc[train_df["split"]>"0915", 'day'] = 10
    del train_df["split"]
    
    train_df['hour'] = train_df['date'].dt.hour
    train_df['minute'] = train_df['date'].dt.minute
    print('runtime: {:3.3f} mins'.format((time.time()-t)/60))
    
    print('=============================================== read test ===============================================')
    test_df = pd.read_pickle("data/test.pickle")
    test_df['date'] = pd.to_datetime(
        test_df['ts'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
    )
    test_df['day'] = 11
    test_df['hour'] = test_df['date'].dt.hour
    test_df['minute'] = test_df['date'].dt.minute
    train_df["istrain"] = 1
    test_df["istrain"] = 0
    df = pd.concat([train_df, test_df], axis=0, ignore_index=False)
    del train_df, test_df, df['date']
    gc.collect()
    print('runtime: {:3.3f} mins'.format((time.time()-t)/60))
    
    print('=============================================== cate enc ===============================================')
    df['lng_lat'] = df['lng'].astype('str') + '_' + df['lat'].astype('str')
    sort_df = df.sort_values('ts').reset_index(drop=True)
    cate_cols = [
        'deviceid', 'newsid', 'pos', 'app_version', 'device_vendor',
        'netmodel', 'osversion', 'device_version', 'lng', 'lat', 'lng_lat'
    ]
    for f in cate_cols:
        print(f)
        map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
        df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
        sort_df[f] = sort_df[f].map(map_dict).fillna(-1).astype('int32')
        df[f + '_count'] = df[f].map(df[f].value_counts())
    del df['guid']
        
    df = reduce_mem(df)
    sort_df = reduce_mem(sort_df)
    print('runtime: {:3.3f} mins'.format((time.time()-t)/60))
        
    print('*************************** exposure_ts_gap ***************************')
    for f in [
        ['deviceid'], ['newsid'], ['lng_lat'],
        ['pos', 'deviceid'], ['pos', 'newsid'], ['pos', 'lng_lat'],
        ['pos', 'deviceid', 'lng_lat'],
        ['netmodel', 'deviceid'], ['pos', 'netmodel', 'deviceid'],
        ['netmodel', 'lng_lat'], ['deviceid', 'lng_lat'],
        ['netmodel', 'deviceid', 'lng_lat'], ['pos', 'netmodel', 'lng_lat'],
        ['pos', 'netmodel', 'deviceid', 'lng_lat']
    ]:
        print('------------------ {} ------------------'.format('_'.join(f)))
        tmp = sort_df[f + ['ts']].groupby(f)
        # 前x次、后x次曝光到当前的时间差
        for gap in [1, 2, 3, 5, 10]:
            sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)] = tmp['ts'].shift(0) - tmp['ts'].shift(gap)
            sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)] = tmp['ts'].shift(-gap) - tmp['ts'].shift(0)
            tmp2 = sort_df[
                f + ['ts', '{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap), '{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
            ].drop_duplicates(f + ['ts']).reset_index(drop=True)
            df = df.merge(tmp2, on=f + ['ts'], how='left')
            del sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)]
            del sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
            del tmp2
        del tmp
        df = reduce_mem(df)
        print('runtime: {:3.3f}'.format(time.time()-t) )
    sort_df.to_pickle("user_data/sort_df.pickle")
    del sort_df
    gc.collect()
    
    for f in [["deviceid"], ['deviceid', 'pos' ], [ 'deviceid', 'lng_lat', 'pos'], [ 'deviceid', 'netmodel', 'pos' ],  ['deviceid', 'netmodel', 'lng_lat', 'pos'] ]:
        df = df.sort_values(f+['ts']).reset_index(drop=True)
        group = df.groupby(f)
        df_c = df.copy()
        df_c['gap_before_1'] = group['ts'].shift(0) - group['ts'].shift(1)
        df_c['gap_after_1'] = group['ts'].shift(-1) - group['ts'].shift(0)
        tmp2 = df_c[f+[ 'ts', 'gap_before_1', 'gap_after_1']].drop_duplicates(f+['ts']).reset_index(drop=True)
        df = df.merge(tmp2, on=f+['ts'], how='left')
        df['gap_before_1'] = df['gap_before_1'].fillna(600 * 1000)
        df['gap_after_1'] = df['gap_after_1'].fillna(600 * 1000)
        del df_c, tmp2
        k = "_".join(f)
        for mins in [0.5,1,3,5]:
            INDEX = df[df['gap_before_1'] > (mins*60*1000-1)].index
            LENGTH = len(INDEX)
            ts_len = []
            for i in range(1, LENGTH):
                ts_len += [(INDEX[i]-INDEX[i - 1])] * (INDEX[i]-INDEX[i - 1])
            ts_len += [(len(df)-INDEX[LENGTH - 1])] * (len(df)-INDEX[LENGTH - 1])
            df['{}_ts_before_len_{}mins'.format(k,mins)] = ts_len
            df['{}_ts_before_rank_{}mins'.format(k,mins)] = group['ts'].apply(lambda x: (x).rank())  
            df['{}_ts_before_rank_{}mins'.format(k,mins)] = (df['{}_ts_before_rank_{}mins'.format(k,mins)] - 1) / (df['{}_ts_before_len_{}mins'.format(k,mins)] - 1)
            
            INDEX = df[df['gap_after_1'] > (mins*60*1000-1)].index
            LENGTH = len(INDEX)
            ts_len = [INDEX[0]] * (INDEX[0]+1)
            for i in range(1, LENGTH):
                ts_len += [(INDEX[i]-INDEX[i - 1])] * (INDEX[i]-INDEX[i - 1])
            df['{}_ts_after_len_{}mins'.format(k,mins)] = ts_len
            df['{}_ts_after_rank_{}mins'.format(k,mins)] = group['ts'].apply(lambda x: (-x).rank())
            df['{}_ts_after_rank_{}mins'.format(k,mins)] = (df['{}_ts_after_rank_{}mins'.format(k,mins)] - 1)/(df['{}_ts_after_len_{}mins'.format(k,mins)] - 1)
            
            df.loc[df['{}_ts_before_rank_{}mins'.format(k,mins)]==np.inf, '{}_ts_before_rank_{}mins'.format(k,mins)] = 0
            df.loc[df['{}_ts_after_rank_{}mins'.format(k,mins)]==np.inf, '{}_ts_after_rank_{}mins'.format(k,mins)] = 0
            df['{}_ts_before_len_{}mins'.format(k,mins)] = np.log(df['{}_ts_before_len_{}mins'.format(k,mins)] + 1)
            df['{}_ts_after_len_{}mins'.format(k,mins)] = np.log(df['{}_ts_after_len_{}mins'.format(k,mins)] + 1)
        print( 'runtime:{:3.3f}'.format(time.time() - t) )
        del group, df['gap_before_1'], df['gap_after_1'] 
        df = reduce_mem(df)
    print('gap_before_1 runtime: {:3.3f}'.format(time.time()-t) )
    
    print('*************************** cross feat (second order) ***************************')
    # 二阶交叉特征，可以继续做更高阶的交叉。
    cross_cols = ['deviceid', 'newsid', 'pos', 'netmodel', 'lng_lat']
    for f in cross_cols:
        for col in cross_cols:
            if col == f:
                continue
            print('------------------ {} {} ------------------'.format(f, col))
            df = df.merge(df[[f, col]].groupby(f, as_index=False)[col].agg({
                'cross_{}_{}_nunique'.format(f, col): 'nunique',
                'cross_{}_{}_ent'.format(f, col): lambda x: entropy(x.value_counts() / x.shape[0]) # 熵
            }), on=f, how='left')
            
            if 'cross_{}_{}_count'.format(f, col) not in df.columns.values and 'cross_{}_{}_count'.format(col, f) not in df.columns.values:
                df = df.merge(df[[f, col, 'id']].groupby([f, col], as_index=False)['id'].agg({
                    'cross_{}_{}_count'.format(f, col): 'count' # 共现次数
                }), on=[f, col], how='left')
                
            if 'cross_{}_{}_count_ratio'.format(col, f) not in df.columns.values:
                df['cross_{}_{}_count_ratio'.format(col, f)] = df['cross_{}_{}_count'.format(f, col)] / (df[f + '_count']+1e-3) # 比例偏好
                
            if 'cross_{}_{}_count_ratio'.format(f, col) not in df.columns.values:
                df['cross_{}_{}_count_ratio'.format(f, col)] = df['cross_{}_{}_count'.format(f, col)] / (df[col + '_count']+1e-3) # 比例偏好
                
            df['cross_{}_{}_nunique_ratio_{}_count'.format(f, col, f)] = df['cross_{}_{}_nunique'.format(f, col)] / (df[f + '_count']+1e-3)
            print('runtime: {:3.3f} mins'.format((time.time()-t)/60))
        df = reduce_mem(df)
    gc.collect()
    df.to_pickle("user_data/cross_feat_df.pickle")
    
def emb(df, f1, f2):
    t = time.time()
    emb_size = 8
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=5, min_count=5, sg=0, hs=1, seed=2019)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model:
                vec.append(model[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix=np.array(emb_matrix)
    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    del model, emb_matrix, sentences
    tmp = reduce_mem(tmp)
    print('runtime: {:3.3f} mins'.format((time.time()-t)/60))
    return tmp

def feat_emb():
    t = time.time()
    df = pd.read_pickle("user_data/cross_feat_df.pickle")
    sort_df = pd.read_pickle("user_data/sort_df.pickle")
    emb_cols = [['deviceid', 'newsid'], ['deviceid', 'lng_lat'], ['newsid', 'lng_lat']]
    for f1, f2 in emb_cols:
        df = df.merge(emb(sort_df, f1, f2), on=f1, how='left')
        df = df.merge(emb(sort_df, f2, f1), on=f2, how='left')
    del sort_df
    gc.collect()
    train_df = df[df["istrain"] == 1]
    test_df = df[df["istrain"] == 0]
    del df, train_df["istrain"], test_df["istrain"]
    gc.collect()
    test_df.to_pickle("user_data/test_df_table.pickle")
    del test_df
    df1 = train_df[train_df["day"]==8]
    df1.to_pickle("user_data/train_day8_table.pickle")
    del df1
    df2 = train_df[train_df["day"]==9]
    df2.to_pickle("user_data/train_day9_table.pickle")
    del df2
    df3 = train_df[train_df["day"]==10]
    df3.to_pickle("user_data/train_day10_table.pickle")
    del train_df, df3
    gc.collect()
    print('feat_emb runtime: {:3.3f} mins'.format((time.time()-t)/60))

def history_feats(featpath,trainpath,writep):
    print("Feature engineer")
    t = time.time()
    train_data = pd.read_pickle(trainpath)
    train_data["istrain"] = 1
    df = pd.read_pickle(featpath)
    df["istrain"] = 0
    train_data = pd.concat([train_data, df], sort=False, ignore_index=True)
    del df
    gc.collect()
    print('concat runtime: {:3.3f}'.format(time.time()-t) )
    
    type_cat = ['deviceid', 'pos', 'netmodel', 'lng_lat', "device_version", "hour", "outertag_key", "applist"]
    train_data[type_cat] = train_data[type_cat].astype(np.str)
    
    union_feature = ["deviceid_netmodel_hour", "deviceid_hour", "deviceid_netmodel","deviceid_pos"] 
    train_data["deviceid_pos"] = train_data["deviceid"] + "_" + train_data["pos"]
    train_data["deviceid_netmodel_hour"] = train_data["deviceid"] + "_" + train_data["netmodel"] + "_" + train_data["hour"]
    train_data["deviceid_netmodel"] = train_data["deviceid"] + "_" + train_data["netmodel"]
    train_data["deviceid_hour"] = train_data["deviceid"] + "_" + train_data["hour"] 
    
    train_data[cate_features] = train_data[cate_features].astype(np.int)
    print('union_feature runtime: {:3.3f}'.format(time.time()-t) )
    
    df = train_data[train_data["istrain"] == 0]
    train_data = train_data[train_data["istrain"] == 1]
    train_data.drop("istrain",axis=1,inplace=True)
    df.drop("istrain",axis=1,inplace=True)
    
    train_data = reduce_mem(train_data)
    count_feature = [ "deviceid", "lng_lat", "device_version"]
    for feature in union_feature+count_feature:
        t = time.time()
        count_table = pd.pivot_table(df,
                                     index=feature,
                                     columns="target",
                                     values="id",
                                     aggfunc="count",
                                     fill_value=0)
        df.drop(feature,axis=1,inplace=True)
        count_table[[1, 0]] = count_table[[1, 0]] + 1
        count_table["count"] = count_table[1] + count_table[0]
        count_table["rate_pos"] = count_table[1] / np.sum(count_table[1]) * 100
        count_table["rate_neg"] = count_table[0] / np.sum(count_table[0]) * 100
        count_table["efficiency"] = count_table["rate_pos"] - count_table["rate_neg"]
        count_table["rate"] = count_table[1] / count_table["count"]
        count_table["woe"] = np.log(count_table["rate_pos"] / count_table["rate_neg"])
        count_table["iv"] = count_table["woe"] * count_table["efficiency"]
        count_table.drop([ 0, "rate_pos", "rate_neg"], axis=1, inplace=True)
        F = []
        for i in count_table.columns:
            if i == 1:F.append(feature +"_count_table_1")
            else:F.append(feature +"_count_table_" + str(i))
        count_table.columns = F
        train_data = train_data.merge(count_table.reset_index(), left_on=feature, right_on=feature, how="left")
        if feature in union_feature:
            train_data.drop(feature,axis=1,inplace=True)
        train_data = reduce_mem(train_data)
        print('count_table runtime: {:3.3f},{}'.format(time.time()-t, feature))
    del count_table,df
    gc.collect()
    train_data.to_pickle(writep)

def user_app():
    user = pd.read_pickle("user_data/user_feature.pickle")
    train_df = pd.read_pickle("data/train.pickle")
    test_df = pd.read_pickle("data/test.pickle")
    df = pd.concat([train_df, test_df], axis=0, ignore_index=False)
    f = 'deviceid'
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    user[f] = user[f].map(map_dict).fillna(-1).astype('int32')
    user[f] = user[f].astype(np.str)
    test_data = pd.read_pickle("user_data/test_11_table.pickle")
    df = df[["id","guid"]].copy()
    test_data = pd.merge(test_data, df, on=['id'], how='left')
    test_data = pd.merge(test_data, user, on=['deviceid','guid'], how='left')
    test_data.to_pickle("user_data/test_11_user.pickle")
    
    test_data = pd.read_pickle("user_data/train_8_table.pickle")
    test_data = pd.merge(test_data, df, on=['id'], how='left')
    test_data = pd.merge(test_data, user, on=['deviceid','guid'], how='left')
    test_data.to_pickle("user_data/train_8_user.pickle")
    
    test_data = pd.read_pickle("user_data/train_10_table.pickle")
    test_data = pd.merge(test_data, df, on=['id'], how='left')
    test_data = pd.merge(test_data, user, on=['deviceid','guid'], how='left')
    test_data.to_pickle("user_data/train_10_user.pickle")
    
    test_data = pd.read_pickle("user_data/train_9_table.pickle")
    test_data = pd.merge(test_data, df, on=['id'], how='left')
    test_data = pd.merge(test_data, user, on=['deviceid','guid'], how='left')
    test_data.to_pickle("user_data/train_9_user.pickle")
     
class Tag_deal():
    def score_mean(self,x):
        score = [] 
        for i in x.split("|"):
            try:s = float(i.split(":")[-1])
            except:continue
            score.append(s)
        if len(score) == 0:
            return np.nan
        return sum(score)/len(score)
    
    def score_max(self,x):
        score = 0
        for i in x.split("|"):
            try:s = float(i.split(":")[-1])
            except:continue
            if s>=score:
                score = s
        if score==0:
            return np.nan
        return score
    
    def score_1(self,x):
        score = 0
        for i in x.split("|"):
            try:s = float(i.split(":")[-1])
            except:continue
            if s>=1:
                score+=1
        return score
    
    def score_quarter_1(self,x):
        score = [] 
        for i in x.split("|"):
            try:s = float(i.split(":")[-1])
            except:continue
            score.append(s)
        if len(score) == 0:
            return np.nan
        score.sort()
        ind = len(score)//4
        return score[ind]
    
    def score_quarter_2(self,x):
        score = [] 
        for i in x.split("|"):
            try: s = float(i.split(":")[-1])
            except: continue
            score.append(s)
        if len(score) == 0:
            return np.nan
        score.sort()
        ind = len(score)//4
        return score[ind*2]
    
    def score_quarter_3(self,x):
        score = [] 
        for i in x.split("|"):
            try: s = float(i.split(":")[-1])
            except: continue
            score.append(s)
        if len(score) == 0:
            return np.nan
        score.sort()
        ind = len(score)//4
        return score[ind*3]
    
    def outertag_key(self,x):
        keys = ['花絮片段', '片段', '社会热点', '表演', '恶搞吐槽', '美食', '娱乐资讯', '时尚', '奇闻轶事', '明星', '吃秀', '真人秀', '情感', '影视', '宠物', '旅行', '穿秀', 
                '明星访谈', '舞蹈', '运动', '音乐', '游戏', '音乐舞蹈', '影评', '自拍', '美妆', '萌宠', '才艺', '宝宝', '高颜值', '探店', '生活', '手工', '画画', '生活百科', 
                'MV', '趣味学堂', '唱歌', '记录', '演唱会', '发型', '母婴', '美食攻略', '雷人囧事', '其他', '健康', '看点历史', '涨姿势', '日韩动漫', '国产动漫', '演奏', '生活小窍门', 
                '微电影', '电子竞技', '美食菜谱', '搞笑', '护理', '明星八卦', '玩具', '摄影摄像', '旅游攻略', '手机游戏']
        bina = 0
        for i in x.split("|"):
            try:s = i.split("_cs:")[0]
            except:continue
            if s in keys:
                bina += 2**keys.index(s)
        return bina**0.15
#         return np.log(bina+1)     
                
    def applist(self,x):
        #最常见的APP
        appset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 19, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 43, 50, 51, 55, 56, 59, 61, 65, 67, 
             69, 70, 71, 73, 79, 83, 84, 85, 86, 87, 88, 89, 90, 95, 96, 97, 106, 109, 131, 133, 135, 141, 148, 158]
        apps = list((map(lambda k:int(k.split("_")[1]),set(x[1:-2].split(" ")))))
        bina = 0
        for ap in apps:
            if ap in appset:
                bina += 2**appset.index(ap)
        return bina**0.15
     
def user_feature():
    T = Tag_deal()
    user = pd.read_csv("data/user.csv",encoding="utf-8")
    print(user.head())
    
    user = reduce_mem(user)
    user[["outertag","tag"]] = user[["outertag","tag"]].astype(np.str)
    user["outertag_key"] = user["outertag"].apply(T.outertag_key)

    user["outertag_count"] = user["outertag"].apply(lambda x:x.count("_cs:"))
    user["tag_count"] = user["tag"].apply(lambda x:x.count("_cs:"))
    
    user["outertag_mean"] = user["outertag"].apply(T.score_mean)
    user["tag_mean"] = user["tag"].apply(T.score_mean)
    
    user["outertag_max"] = user["outertag"].apply(T.score_max)
    user["tag_max"] = user["tag"].apply(T.score_max)
    
    user["outertag_1"] = user["outertag"].apply(T.score_1)
    
    user["tag_quarter_1"] = user["tag"].apply(T.score_quarter_1)
    user["tag_quarter_2"] = user["tag"].apply(T.score_quarter_2)
    user["tag_quarter_3"] = user["tag"].apply(T.score_quarter_3)
    
    app = pd.read_csv("data/app.csv",encoding="utf-8")
    app['appnum'] = app['applist'].apply(lambda x: x.count("app_"))
    app['applist'] = app['applist'].apply(T.applist)
    app = app.groupby("deviceid").agg("mean")
    user = pd.merge(user,app,on=['deviceid'],how='left')
    user.drop(['outertag', 'tag', ],axis=1,inplace=True)
    print(user.head())
    user = reduce_mem(user)
    user.to_pickle("user_data/user_feature.pickle")

def data_deal():
    """特征工程
    """
    user_feature()
    df = pd.read_csv("data/train.csv",encoding="utf-8")
    df = reduce_mem(df)
    df.to_pickle("data/train.pickle")
    leak_feats()
    feat_emb()
    history_feats("user_data/train_day9_table.pickle", "user_data/train_day8_table.pickle", "user_data/train_8_table.pickle")
    history_feats("user_data/train_day8_table.pickle", "user_data/train_day9_table.pickle", "user_data/train_9_table.pickle")
    history_feats("user_data/train_day9_table.pickle", "user_data/train_day10_table.pickle", "user_data/train_10_table.pickle")
    history_feats("user_data/train_day10_table.pickle", "user_data/test_df_table.pickle", "user_data/test_11_table.pickle")
    user_app()
      
def click_dataset(ids,istest=False):
    rows = ["id", "timestamp", "target", "day", 'newsid', 'lng', 'lat', 'deviceid', 'lng_lat', "device_version"]
    rows += ['cross_netmodel_pos_ent', 'cross_pos_lng_lat_nunique', 'cross_netmodel_pos_nunique', 'cross_netmodel_lng_lat_nunique', 'cross_pos_netmodel_nunique', 'cross_netmodel_newsid_ent']
    rows += ['cross_netmodel_lng_lat_nunique_ratio_netmodel_count', 'cross_pos_newsid_nunique', 'cross_pos_newsid_nunique_ratio_pos_count','cross_netmodel_pos_nunique_ratio_netmodel_count','cross_pos_netmodel_nunique_ratio_pos_count','cross_netmodel_newsid_nunique', 'cross_pos_lng_lat_nunique_ratio_pos_count']
    rows += ["ts",'personidentification', 'gender', "guid"]
    if istest:
        test_data = pd.read_pickle("user_data/test_11_user.pickle")
        test_data["day_minute"] = test_data["hour"]*60 + test_data["minute"]
        sub = test_data[["id"]].copy()
        for col in test_data.columns:
            if "app" in col and col != "app_version":
                rows.append(col)
            if "outertag_key" in col :
                rows.append(col)
        test_data.drop(rows,axis=1,inplace=True)
        return sub, test_data
    day8 = pd.read_pickle("user_data/train_8_user.pickle")
    day9 = pd.read_pickle("user_data/train_9_user.pickle")
    day10 = pd.read_pickle("user_data/train_10_user.pickle")
    if ids == 1:
        vailddata = day9
        traindata = pd.concat([day10, day8], sort=False, ignore_index=True)
        del day8,day10
    elif ids == 2:
        vailddata = day10
        traindata = pd.concat([day9, day8], sort=False, ignore_index=True)
        del day8, day9
    else:
        vailddata = day8
        traindata = pd.concat([day9, day10], sort=False, ignore_index=True)
        del day9, day10
    traindata["day_minute"] = traindata["hour"]*60 + traindata["minute"]
    vailddata["day_minute"] = vailddata["hour"]*60 + vailddata["minute"]
    for col in traindata.columns:
        if "app" in col and col != "app_version":
            rows.append(col)
        if "outertag_key" in col :
            rows.append(col)
    train_y = traindata["target"].copy()
    vaild_y = vailddata["target"].copy()
    traindata.drop(rows,axis=1,inplace=True)
    vailddata.drop(rows,axis=1,inplace=True)
    return traindata, train_y, vailddata, vaild_y

def train_vaild(ids):
    global params
    print("prepare load train data.......")
    t = time.time()
    traindata, train_y, vailddata, vaild_y = click_dataset(ids)
    lgb_train = lgb.Dataset(traindata, train_y, categorical_feature=cate_features, silent=True)
    lgb_eval = lgb.Dataset(vailddata, vaild_y, reference=lgb_train, categorical_feature=cate_features, silent=True)
    del train_y,traindata
    gc.collect()
    print("Memory free: {:2.4f} GB".format(psutil.virtual_memory().free / (1024**3)))
    print( 'runtime:{:3.3f}'.format(time.time() - t) )
    gbm = lgb.train(params, lgb_train, num_boost_round=50000, valid_sets=[lgb_train,lgb_eval], verbose_eval=500, early_stopping_rounds=500)
    gbm.save_model('save_model/gap_model_{}.cpt'.format(ids), num_iteration=gbm.best_iteration)

    vaild_preds = gbm.predict(vailddata, num_iteration=gbm.best_iteration)
    vaild_preds = MinMaxScaler().fit_transform(vaild_preds.reshape(-1, 1))
    vaild_preds = vaild_preds.reshape(-1, )
    
    d = {'pred': vaild_preds, 'real': vaild_y}
    result = pd.DataFrame(data=d)
    result.to_csv("result_sub/gap_vaild_test_{}.csv".format(ids),index=None,encoding='utf-8')
    
    threshold_test = round(np.percentile(vaild_preds, P), 4)
    vaild_preds = vaild_preds>threshold_test
    ff = f1_score(vaild_y,vaild_preds)
    if not os.path.exists("feature"):
        os.mkdir("feature")
    importance = gbm.feature_importance()
    feature_name = gbm.feature_name()
    feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} ).sort_values(by='importance',ascending=False)
    feature_importance.to_csv('feature/feature_importance_{:1.5f}.csv'.format(ff),index=None,encoding='utf-8-sig')
    print("#F1 score:{:2.5f}".format(ff))
    return gbm.best_iteration

def train(ids, num_boost):
    global params
    print("prepare load train data.......")
    t = time.time()
    traindata, train_y, vailddata, vaild_y = click_dataset(ids)
    traindata = pd.concat([traindata, vailddata], sort=False, ignore_index=True)
    train_y = pd.concat([train_y, vaild_y], sort=False, ignore_index=True)
    
    lgb_train = lgb.Dataset(traindata, train_y, categorical_feature=cate_features, silent=True)
    del train_y,traindata
    gc.collect()
    print("Memory free: {:2.4f} GB".format(psutil.virtual_memory().free / (1024**3)))
    print( 'runtime:{:3.3f}'.format(time.time() - t) )
    gbm = lgb.train(params, lgb_train, num_boost_round=num_boost, valid_sets=[lgb_train], verbose_eval=500, early_stopping_rounds=500)
    gbm.save_model('save_model/gap_model_{}.cpt'.format(ids), num_iteration=gbm.best_iteration)

    importance = gbm.feature_importance()
    feature_name = gbm.feature_name()
    feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} ).sort_values(by='importance',ascending=False)
    feature_importance.to_csv('feature/gap_feature_importance_{:1.4f}.csv'.format(0.9),index=None,encoding='utf-8-sig')
    return gbm.best_iteration
    
def predict(ids):
    print("prepare load predict data.......")
    sub, test_data = click_dataset(ids,istest=True)
    gbm = lgb.Booster(model_file='save_model/gap_model_{}.cpt'.format(ids))
    test_pre = gbm.predict(test_data, num_iteration=gbm.best_iteration)
    test_pre = MinMaxScaler().fit_transform(test_pre.reshape(-1, 1))
    test_pre = test_pre.reshape(-1, )
    sub["Label"] = test_pre
    sub.to_csv("result_sub/submission_test_{}.csv".format(ids), index=False,encoding='utf-8') 

def score_vail(its):
    import matplotlib.pylab as plt
    re = {}
    plt.figure(figsize=(16,5*10))
    for idx in range(its):
        score = []
        sub = pd.read_csv("result_sub/his_vaild_test_{}.csv".format(idx+1),encoding="utf-8")
        t = round(np.percentile(sub["pred"], 89), 4)
        print(t)        
        cd = dict(sub["real"].value_counts())
        s = 0
        bp = 0
        for i in range(150):
            p = 88.2+i*0.01
            threshold_test = round(np.percentile(sub["pred"], p), 4)
            sub["test"] = sub["pred"]>threshold_test
            ff = f1_score(sub["real"],sub["test"])
            score.append(ff)
            plt.plot(range(len(score)), score)
            if ff>=s:
                s = ff
                bp = p
        re[idx] = ["result_sub/vaild_test_{}.csv".format(idx+1), "F1 score:{:2.4f} ".format(s),"best P:{:2.2f} ".format(bp),cd,cd[0]/(cd[1]+cd[0])]
        print(re[idx])
    print(re)
    plt.show()

def submission(its):
    sub = pd.read_csv("submission_test_4.csv",encoding="utf-8")
    for i in range(1,its):
        sub1 = pd.read_csv("submission_test_{}.csv".format(i+1),encoding="utf-8")
        sub1.rename(columns={"Label":"Label_{}".format(i+1)},inplace=True)
        sub = pd.merge(sub,sub1,on=['id'],how='right')
    if its > 2:
        sub["Label"] = (sub["Label"]  +  sub["Label_2"] +  sub["Label_3"])/3 + sub["Label_4"]
    print(sub[["id","Label"]].head())
    sub[["id","Label"]].to_csv('sub_pro.csv', index=False,encoding='utf-8') 
    threshold_test = round(np.percentile(sub["Label"], P), 4)
    print(threshold_test)
    sub["target"] = sub["Label"] > threshold_test
    sub["target"]=sub["target"].astype('int')
    sub = reduce_mem(sub)
    sub[["id","target"]].to_csv('sub_com.csv', index=False,encoding='utf-8') 
    print(sub.head())


if __name__ == "__main__":
    print("start")
    P = 89
    params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 255,
    'max_bin': 425,
    "learning_rate":0.1,
    "colsample_bytree":0.8,#每次迭代中随机选择特征的比例
    "bagging_fraction":0.9, #每次迭代时用的数据比例
    'min_child_samples': 10,
    'reg_alpha':3, 
    'reg_lambda':0.1,
    'n_jobs': -1,
    'seed':1000,
    }
    a = time.time()
    print("start time {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    data_deal() #特征工程
    num_boost = 0
    for i in range(3):
        #三折训练，以及获得三次训练的最佳迭代次数
        num_boost += train_vaild(i+1) 
        
    #使用全量数据训练，由于没有验证集，需要合理制定最大迭代次数
    train(4, num_boost//3)
    for i in range(4):
        predict(i+1)
    submission(4)

    print("{} - time:{:6.4f} mins".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (time.time()-a)/60))



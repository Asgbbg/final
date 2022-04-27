import gensim
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Logloss import multiclass_logloss

data = pd.read_excel('data/fudan-preprocess.xlsx')
X = data['文本分词']
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.分类.values)
test_size = 0.3
xtrain, xvalid, ytrain, yvalid = train_test_split(data.文本分词.values, y,
                                                  stratify=y,
                                                  random_state=41,
                                                  test_size=test_size, shuffle=True)
print(xtrain.shape)
print(xvalid.shape)
X = [i.split() for i in X]
model = gensim.models.Word2Vec(X, min_count=5, window=8, vector_size=100)  # X是经分词后的文本构成的list，也就是tokens的列表的列表
embeddings_index = dict(zip(model.wv.index_to_key, model.wv.vectors))
# X是经分词后的文本构成的list，也就是tokens的列表的列表。
# 注意，Word2Vec还有3个值得关注的参数，iter是模型训练时迭代的次数，假如参与训练的文本量较少，就需要把这个参数调大一些；sg是模型训练算法的类别，
# 1 代表 skip-gram，;0代表 CBOW;window控制窗口，它指当前词和预测词之间的最大距离，
# 如果设得较小，那么模型学习到的是词汇间的功能性特征（词性相异），如果设置得较大，会学习到词汇之间的相似性特征（词性相同）的大小。
print('Found %s word vectors.' % len(embeddings_index))


def sent2vec(s):
    import jieba
    stwlist = [line.strip() for line in open('dicts/stopwords.txt', 'r', encoding='utf-8').readlines()]
    words = str(s).lower()
    words = jieba.lcut(words)
    words = [w for w in words if not w in stwlist]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


print('开始xtrain句子to词向量')
xtrain_w2v = [sent2vec(x) for x in tqdm(xtrain)]
print('开始xvalid句子to词向量')
xvalid_w2v = [sent2vec(x) for x in tqdm(xvalid)]

print("保存模型中")
joblib.dump(xtrain_w2v, 'models/xtrain_w2v.pkl')
joblib.dump(xvalid_w2v, 'models/xvalid_w2v.pkl')
print("保存成功")

xtrain_w2v = np.array(xtrain_w2v)
xvalid_w2v = np.array(xvalid_w2v)

print("XGB拟合")
clf = xgb.XGBClassifier(nthread=10, silent=False)
clf.fit(xtrain_w2v, ytrain)
predictions = clf.predict_proba(xvalid_w2v)

try:
    print(classification_report(predictions, yvalid))
except Exception as e:
    print("出现了异常" + str(e))
else:
    predictions_process = []
    for i in range(len(predictions)):
        max_value = max(predictions[i])
        for j in range(len(predictions[i])):
            if max_value == predictions[i][j]:  # 如果概率最符合该主题，则返回该主题的id
                predictions_process.append(int(j))
print(classification_report(predictions_process, yvalid))
print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

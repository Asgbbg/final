import os
from datetime import datetime

import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, decomposition
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from Logloss import multiclass_logloss
from preprocess import NumberNormalizingVectorizer

curr_time = datetime.now()
time_str = datetime.strftime(curr_time, '%Y-%m-%d-%H-%M-%S')
flag = time_str
# 注意TI-IDF或CountVectorizer只有一个为1
CountVectorizerFlag = 0
TFIDFFlag = 1
# logisticRegression或NaiveBayesFlag只有一个为1
LogisticRegressionFlag = 0
NaiveBayesFlag = 0
SVMFlag = 1
XGBoostFlag = 1


def savelog(report, logloss, tfvprofile, test_size, clf):
    os.makedirs('res/' + flag)
    fileObject = open('res/' + flag + '/res.txt', mode='w')
    fileObject.write('CountVectorizerFlag:' + str(CountVectorizerFlag) + '|')
    fileObject.write('TFIDFFlag:' + str(TFIDFFlag) + '|')
    fileObject.write('LogisticRegressionFlag:' + str(LogisticRegressionFlag) + '|')
    fileObject.write('SVMFlag:' + str(SVMFlag) + '|')
    fileObject.write('XGBoostFlag:' + str(XGBoostFlag) + '|')
    fileObject.write('NaiveBayesFlag:' + str(NaiveBayesFlag) + '\n')
    fileObject.write('testsize:' + str(test_size) + '\n')
    fileObject.write('report:' + str(report) + '\n')
    fileObject.write('logloss:' + str(logloss) + '\n')
    fileObject.write('tfvprofile:' + str(tfvprofile) + '\n')
    fileObject.write('clf:' + str(clf) + '\n')
    fileObject.close()

    # 使用Count Vectorizer来fit训练集和测试集（半监督学习）


def count_vector(xtrain, xvalid):
    ctv = CountVectorizer(min_df=3,
                          max_df=0.5,
                          ngram_range=(1, 2),
                          stop_words=stwlist)
    print("使用CountVectorizer")
    print('ctv参数为:' + str(ctv))
    ctv.fit(list(xtrain) + list(xvalid))
    xtrain_ctv = ctv.transform(xtrain)
    xvalid_ctv = ctv.transform(xvalid)
    return xtrain_ctv, xvalid_ctv


# 使用TF-IDF来fit训练集和测试集（半监督学习）
def number_normalizing_vector(xtrain, xvalid):
    tfv = NumberNormalizingVectorizer(min_df=3,
                                      max_df=0.5,
                                      max_features=None,
                                      ngram_range=(1, 2),
                                      use_idf=True,
                                      smooth_idf=True,
                                      stop_words=stwlist)
    print("使用TF-IDF")
    print("TFV参数为:" + str(tfv))
    tfv.fit(list(xtrain) + list(xvalid))
    xtrain_tfv = tfv.transform(xtrain)
    xvalid_tfv = tfv.transform(xvalid)
    return xtrain_tfv, xvalid_tfv


# 利用提取的TFIDF特征来fit一个简单的Logistic Regression
def logistic_regression(xtrain, ytrain):
    print("使用Logistic Regression")
    clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
    print('Logistic Regressionc参数为:' + str(clf))
    clf.fit(xtrain, ytrain)
    return clf


# 朴素贝叶斯
def naive_bayes(xtrain, ytrain):
    print("使用NaiveBayes")
    clf = MultinomialNB()
    print('NaiveBayes参数为:' + str(clf))
    clf.fit(xtrain, ytrain)
    return clf


def svd_process(xtrain, xvalid):
    # TFV needed
    # 使用SVD进行降维，components设为120，对于SVM来说，SVD的components的合适调整区间一般为120~200
    print("使用SVD进行降维")
    svd = decomposition.TruncatedSVD(n_components=120)
    print('svd参数为:' + str(svd))
    svd.fit(xtrain)
    xtrain_svd = svd.transform(xtrain)
    xvalid_svd = svd.transform(xvalid)

    # 对从SVD获得的数据进行缩放
    print("对从SVD获得的数据进行缩放")
    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    xvalid_svd_scl = scl.transform(xvalid_svd)
    return xtrain_svd_scl, xvalid_svd_scl


def svm_model(xtrain, ytrain):
    # 调用下SVM模型
    print("调用SVM模型")
    clf = SVC(C=1.0, probability=True)  # since we need probabilities
    print("SVM模型参数为:" + str(clf))
    clf.fit(xtrain, ytrain)


def xgboost(xtrain, ytrain):
    print("使用XGB")
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1, silent=False)
    print("XGB参数为:" + str(clf))
    clf.fit(xtrain, ytrain)
    return clf


if __name__ == '__main__':
    data = pd.read_excel('data/fudan-preprocess.xlsx')
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(data.分类.values)
    test_size = 0.3
    xtrain, xvalid, ytrain, yvalid = train_test_split(data.文本分词.values, y,
                                                      stratify=y,
                                                      random_state=41,
                                                      test_size=test_size, shuffle=True)
    print(xtrain.shape)
    print(xvalid.shape)
    stwlist = [line.strip() for line in open('dicts/stopwords.txt', 'r', encoding='utf-8').readlines()]
    xtrain_tfv, xvalid_tfv = number_normalizing_vector(xtrain, xvalid)
    xtrain_svd_scl, xvalid_svd_scl = svd_process()
    clf = xgboost(xtrain_svd_scl, ytrain)
    predictions = clf.predict_proba(xvalid_svd_scl)
    # predictions中的格式为
    # [[0. 0. 0. ... 0. 1. 0.]
    #  [0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 1. 0. 0.]
    #  ...
    #  [0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 0. 1. 0.]]
    # 每行为一个数据，每列为一个可能的tag，数值代表可能是该tag的概率
    # yvalid中为[17 13 16 ... 10 10 17]，代表一个最有可能的tag
    # classification_report函数输入独热编码，即1 0，需要进行一个转换
    # ---------------------------------------------------------------
    predictions_process = []
    for i in range(len(predictions)):
        max_value = max(predictions[i])
        for j in range(len(predictions[i])):
            if max_value == predictions[i][j]:  # 如果概率最符合该主题，则返回该主题的id
                predictions_process.append(int(j))

    # ---------------------------------------------------------------
    report = classification_report(yvalid, predictions_process)
    print(report)
    print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    # savelog(report,multiclass_logloss(yvalid,predictions),tfv,test_size,clf)

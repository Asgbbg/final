import os
import time
from datetime import datetime

import jieba.posseg as psg
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from preprocess import totaltextpre

curr_time = datetime.now()
time_str = datetime.strftime(curr_time, '%Y-%m-%d-%H-%M-%S')
flag = time_str
os.makedirs('models/' + flag)


def readpretxt(filepath, encoding='utf-8'):
    list1 = []
    for line in open(filepath, mode='r', encoding=encoding):
        list1.append(line)
    return list1


def readtxt(filepath, encoding='utf-8'):
    words = [line.strip() for line in open(filepath, mode='r', encoding=encoding).readlines()]
    return words


# 调用函数

def readcsv(filepath, encoding='gbk'):
    data = pd.read_csv(filepath, encoding=encoding)
    text = totaltextpre()
    return text


def cut_word(text):
    # 加载用户自定义词典
    # jieba.load_userdict("./user_dict.txt")
    stopwords = readtxt('dicts/stopwords.txt', )
    sentence = ""
    checkarr = ['n']
    for word, flag in psg.lcut(text):
        if (flag in checkarr) and (word not in stopwords) and (len(word) > 1):
            sentence = sentence + word + " "
    return sentence


if __name__ == '__main__':
    # text_list = totaltextpre()
    print("# 剪切特征关键词")
    # segged_words = [cut_word(x) for x in text_list]
    # print(type(segged_words))
    segged_words1 = readpretxt('data/preprocess.txt')
    n_features = 2000  # 指定特征关键词提取最大值
    print("正在提取词频")
    cv = CountVectorizer(strip_accents='unicode',  # 将使用unicode编码在预处理步骤去除raw document中的重音符号
                         max_features=n_features,
                         max_df=0.5,  # 阈值如果某个词的document frequence大于max_df，不当作关键词
                         min_df=5  # 如果某个词的document frequence小于min_df，则这个词不会被当作关键词
                         )
    tf = cv.fit_transform(segged_words1)
    # 查看构建的词典
    # print(cv.vocabulary_)
    # print(len(cv.vocabulary_))
    # print(len(cv.get_feature_names_out())) # Please use get_feature_names_out instead.
    # 说明：'研究成果': 67,表示'研究成果'这个词在词典中的索引为67
    # print(tf.toarray()[0])
    # print(len(tf.toarray()[0]))
    # print(len(tf.toarray().sum(axis=0)))
    print("# （1）获取高频词的索引")
    fre = tf.toarray().sum(axis=0)
    index_lst = []
    for i in range(len(fre)):
        if fre[i] > 10:  # 词频大于10的定义为高频词
            index_lst.append(i)

    print("# （2）对词典按词频升序排序")
    voca = list(cv.vocabulary_.items())
    sorted_voca = sorted(voca, key=lambda x: x[1], reverse=False)

    print("# （3）提取高频词")
    high_fre_vaca = []
    for i in sorted_voca:
        if i[1] in index_lst:
            high_fre_vaca.append(i[0])

    fileObject = open('models/high_fre_vaca.txt', mode='w')
    for words in high_fre_vaca:
        fileObject.write(words)
        fileObject.write('\n')
    fileObject.close()

    # 模型初始化
    print("开始LDA模型构造")
    k = 150  # 人为指定划分的主题数k
    lda = LatentDirichletAllocation(n_components=k,
                                    max_iter=50,
                                    learning_method='online',
                                    learning_offset=50,
                                    random_state=0)
    time_start = time.time()
    ldamodel = lda.fit_transform(tf)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    proba = np.array(ldamodel)
    print('每个文摘属于各个主题的概率:\n', proba)
    # 构建一个零矩阵
    zero_matrix = np.zeros([proba.shape[0]])
    # 对比所属两个概率的大小，确定属于的类别
    max_proba = np.argmax(proba, axis=1)  # 返回沿轴axis最大值的索引，axis=1代表行；最大索引即表示最可能表示的数字是多少
    print('每个文档所属类别：', max_proba)
    print(len(max_proba))

    weight_matrix = lda.components_
    tf_feature_names = cv.get_feature_names_out()
    id = 0
    file1 = open('models/' + flag + '/100topics100wors.txt', mode='w', encoding='utf-8')
    for weights in weight_matrix:
        dicts = [(name, weight) for name, weight in zip(tf_feature_names, weights)]
        dicts = sorted(dicts, key=lambda x: x[1], reverse=True)  # 根据特征词的权重降序排列
        dicts = [word for word in dicts if word[1] > 0.06]  # 打印权重值大于0.06的主题词
        dicts = dicts[:20]  # 打印每个主题前20个主题词
        # print('主题%d:' % (id), dicts1)
        print([x for x, _ in dicts])
        file1.write(str([x for x, _ in dicts]) + '\n')
        # file1.write('\n')
        id += 1

    print("#接下来保存模型")
    joblib.dump(cv.vocabulary_, 'models/' + flag + '/cv_dic.pkl')
    joblib.dump(tf, 'models/' + flag + '/tf.pkl')
    joblib.dump(ldamodel, 'models/' + flag + '/lda-model.pkl')
    joblib.dump(weight_matrix, 'models/' + flag + '/weight_matrix.pkl')
    joblib.dump(cv, 'models/' + flag + '/cv_profile.pkl')
    joblib.dump(lda, 'models/' + flag + '/lda_profile.pkl')
    fileObject = open('models/' + flag + '/high_fre_vaca.txt', mode='w')
    for words in high_fre_vaca:
        fileObject.write(words)
        fileObject.write('\n')
    fileObject.close()

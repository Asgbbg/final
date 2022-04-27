import re

import jieba
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from Logloss import multiclass_logloss


def readtxt(filepath, encoding='utf-8'):
    words = [line.strip() for line in open(filepath, mode='r', encoding=encoding).readlines()]
    return words


def cn(text):
    str = re.findall(u"[\u4e00-\u9fa5]+", text)
    return str


with open('dicts/stopwords.txt', encoding='utf-8') as f:
    stopwords = [s.strip() for s in f.readlines()]


    # def cut_word_save(text):
    #     stopwords = readtxt('dicts/stopwords.txt', )
    #     sentence = ""
    #     checkarr = ['n']
    #     for word, flag in psg.lcut(text):
    #         if (flag in checkarr) and (word not in stopwords) and (len(word) > 1):
    #             sentence = sentence + word + " "
    #     fileObject1.write(sentence)
    #     fileObject1.write('\n')

    def ldajiebacut(text):
        text = str(cn(text)).replace('', "")
        # jieba.load_userdict('')
        words = [w for w in jieba.lcut(text) if w not in stopwords]
        print(words, len(words))
        words = [w for w in words if len(words) > 2]
        return words


    def totaltextpre():
        print("正在进行总文档的提取，共1000000条")
        list = []
        i = 0
        for eachline in open('SMPCUP2017数据集/1_BlogContent.txt', mode='r', encoding='utf-8'):
            blog_id = eachline.split('\001')[0]
            blog_title = eachline.split('\001')[1]
            blog_content = eachline.split('\001')[2]
            sentence = blog_title + blog_content
            list.insert(i, sentence)
            i = i + 1
            if i == 80000:
                break
        return list


    def number_normalizer(tokens):
        """ 将所有数字标记映射为一个占位符（Placeholder）。
        对于许多实际应用场景来说，以数字开头的tokens不是很有用，
        但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
        """
        return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


    class NumberNormalizingVectorizer(TfidfVectorizer):
        def build_tokenizer(self):
            tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
            return lambda doc: list(number_normalizer(tokenize(doc)))

if __name__ == '__main__':
    # fileObject1 = open('data/preprocess.txt', mode='w', encoding='utf-8')
    # text_list = totaltextpre()
    # print("开始进行分词")
    # i = 0
    # for x in text_list:
    #     print("进度为：" + str(i / 800) + "%")
    #     cut_word_save(x)
    #     i = i + 1
    # fileObject1.close()

    data = pd.read_excel('data/fudan-preprocess.xlsx')
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(data.分类.values)
    xtrain, xvalid, ytrain, yvalid = train_test_split(data.文本分词.values, y,
                                                      stratify=y,
                                                      random_state=42,
                                                      test_size=0.1, shuffle=True)
    print(xtrain.shape)
    print(xvalid.shape)
    stwlist = [line.strip() for line in open('dicts/stopwords.txt',
                                             'r', encoding='utf-8').readlines()]
    tfv = NumberNormalizingVectorizer(min_df=3,
                                      max_df=0.5,
                                      max_features=None,
                                      ngram_range=(1, 2),
                                      use_idf=True,
                                      smooth_idf=True,
                                      stop_words=stwlist)

    # 使用TF-IDF来fit训练集和测试集（半监督学习）
    tfv.fit(list(xtrain) + list(xvalid))
    xtrain_tfv = tfv.transform(xtrain)
    xvalid_tfv = tfv.transform(xvalid)
    clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
    clf.fit(xtrain_tfv, ytrain)
    predictions = clf.predict_proba(xvalid_tfv)
    print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

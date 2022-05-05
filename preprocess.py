import re

import jieba
import pandas as pd
import thulac
from sklearn.feature_extraction.text import TfidfVectorizer


def readtxt(filepath, encoding='utf-8'):
    words = [line.strip() for line in open(filepath, mode='r', encoding=encoding).readlines()]
    return words


def cn(text):
    str = re.findall(u"[\u4e00-\u9fa5]+", text)
    return str


with open('dicts/stopwords.txt', encoding='utf-8') as f:
    stopwords = [s.strip() for s in f.readlines()]


    def textlessthan(sourceurl, targeturl):
        df = pd.read_excel(sourceurl)
        df['text'] = df['text'].astype('str')
        mask = (df['text'].str.len() > 100)
        df.loc[mask]
        df.to_excel(targeturl)


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


    def text_duplicate(sourceurl, targeturl):
        data = pd.read_excel(sourceurl)
        data.drop_duplicates(subset=['tags', 'text'], keep='first', inplace=True)
        data.to_excel(targeturl)
        data.close()


    def number_normalizer(tokens):
        """ 将所有数字标记映射为一个占位符（Placeholder）。
        对于许多实际应用场景来说，以数字开头的tokens不是很有用，
        但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
        """
        return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


    def fencibyjieba(sourceurl):
        data = pd.read_excel(sourceurl)
        # print(data.tags.unique())
        data['文本分词'] = data['text'].apply(lambda i: jieba.cut(i))
        data['文本分词'] = [' '.join(i) for i in data['文本分词']]
        data.to_excel('data/total_process-new.xlsx')


    def fencibythulac(sourceurl):
        thucut = thulac.thulac(seg_only=True)
        data = pd.read_excel(sourceurl)
        data['文本分词'] = data['text'].apply(lambda i: thucut.cut(i))
        # data['文本分词'] = [' '.join(i) for i in data['文本分词']]
        df = pd.DataFrame[data['文本分词']]
        df.to_excel('total_process-new.xlsx')


    class NumberNormalizingVectorizer(TfidfVectorizer):
        def build_tokenizer(self):
            tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
            return lambda doc: list(number_normalizer(tokenize(doc)))

if __name__ == '__main__':
    # text_duplicate('data/TOTAL-new-副本.xlsx','data/TOTAL_QUCHONG.xlsx')
    # 20888行去重前
    # textlessthan('data/爬虫/TOTAL_QUCHONG.xlsx', 'data/total_process.xlsx')
    fencibyjieba('data/total_process_more100less1000.xlsx')
    # fencibythulac('data/total_process_more50.xlsx')

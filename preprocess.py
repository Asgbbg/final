import re

import jieba
import jieba.posseg as psg


def readtxt(filepath, encoding='utf-8'):
    words = [line.strip() for line in open(filepath, mode='r', encoding=encoding).readlines()]
    return words


def cn(text):
    str = re.findall(u"[\u4e00-\u9fa5]+", text)
    return str


with open('dicts/stopwords.txt', encoding='utf-8') as f:
    stopwords = [s.strip() for s in f.readlines()]


    def cut_word_save(text):
        # 加载用户自定义词典
        # jieba.load_userdict("./user_dict.txt")
        stopwords = readtxt('dicts/stopwords.txt', )
        sentence = ""
        checkarr = ['n']
        for word, flag in psg.lcut(text):
            if (flag in checkarr) and (word not in stopwords) and (len(word) > 1):
                sentence = sentence + word + " "
        fileObject1.write(sentence)
        fileObject1.write('\n')


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

if __name__ == '__main__':
    fileObject1 = open('data/preprocess.txt', mode='w', encoding='utf-8')
    text_list = totaltextpre()
    print("开始进行分词")
    i = 0
    for x in text_list:
        print("进度为：" + str(i / 800) + "%")
        cut_word_save(x)
        i = i + 1
    fileObject1.close()

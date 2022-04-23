# 将训练集按8:2分为训练集和验证集
import csv

file_add = 'train/SMPCUP2017_TrainingData_Task1.txt'
csvnametrain = 'trainingDATA-800.csv'
csvnamevalidation = 'validationDATA-200'
i = 0
csvfile1 = open(csvnametrain, mode='w+', encoding='utf-8', newline='')
csvfile2 = open(csvnamevalidation, mode='w+', encoding='utf-8', newline='')
for lines in open(file_add, encoding="utf-8"):
    i = i + 1
    blog_id_answer = lines.strip().split('\001')[0]
    # 验证集为220左右
    if i > 800:
        # f_answer = list(open(file_add, encoding="utf-8").readlines())
        # print(blog_id_answer,i)
        writer = csv.writer(csvfile2)
        writer.writerow([blog_id_answer])
    # 训练集为800个
    else:
        writer = csv.writer(csvfile1)
        writer.writerow([blog_id_answer])

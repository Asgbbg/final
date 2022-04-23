# 将训练集和验证集根据ID，从数据集中选取出相应的文章，方便以后做训练和验证
import csv

train_div = 'data/trainingDATA-800.csv'
validation_div = 'data/validationDATA-200.csv'
csv_train_text = 'data/trainingTEXT-800.csv'
csv_validation_text = 'data/validationTEXT-200.csv'
total_data = 'SMPCUP2017数据集/1_BlogContent.txt'
# 打开训练集文本
file_train_text = open(csv_train_text, mode='w', encoding='utf-8', newline='')
writer1 = csv.writer(file_train_text)
# 打开验证集文本
file_val_text = open(csv_validation_text, mode='w', encoding='utf-8', newline='')
writer2 = csv.writer(file_val_text)


def train(id, title, text):
    for train_lines in open(train_div, mode='r'):
        train_id = train_lines.strip().split('\001')[0]
        if train_id == id:
            writer1.writerow([id, title, text])


def val(id, title, text):
    for val_lines in open(validation_div, mode='r'):
        val_id = val_lines.strip().split('\001')[0]
        if val_id == id:
            # print(title, text)
            writer2.writerow([id, title, text])


if __name__ == '__main__':
    i = 0
    for total_lines in open(total_data, encoding='utf-8', mode='r'):
        i = i + 1
        blog_text_id = total_lines.strip().split('\001')[0]
        blog_title = total_lines.strip().split('\001')[1]
        blog_text = total_lines.strip().split('\001')[2]
        train(blog_text_id, blog_title, blog_text)
        val(blog_text_id, blog_title, blog_text)
        if (i % 10000) == 0:
            print('目前进度为' + str(i / 10000))

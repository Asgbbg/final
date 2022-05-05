import numpy as np
import pandas as pd
import sklearn
from simpletransformers.classification import ClassificationModel
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_excel(r'data/total_process-newmore100less1000byjieba.xlsx')
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.tags.values)
data['tags2num'] = data['tags'].apply(lambda x: lbl_enc.transform([x])[0])
data = data[['文本分词', 'tags2num']]
# print(data)
# print(y,lbl_enc.inverse_transform([1,2,3,4,5]))
# data[4]=data[0].apply(lambda x:lbl_enc.transform([x])[0])

num_labels = len(np.unique(y.tolist()))
xtrain, xvalid, ytrain, yvalid = train_test_split(data, y, stratify=y,
                                                  test_size=0.1, shuffle=True)
# print(num_labels)
# print(xtrain)
# config = BertConfig.from_json_file('models/chinese_L-12_H-768_A-12/config.json')
# config.num_labels=num_labels
if __name__ == '__main__':
    # print(torch.cuda.is_available())
    # print(torch.version.cuda)
    model = ClassificationModel(model_type='bert',
                                model_name='models/chinese-bert-wwm-ext',
                                num_labels=num_labels, use_cuda=True,
                                args={"reprocess_input_data": True,  # 对输入数据进行预处理
                                      "overwrite_output_dir": True,
                                      "num_train_epochs": 30,

                                      "evaluate_during_training_verbose": True,
                                      "evaluate_during_training_steps": 1000,
                                      "evaluate_during_training_verbose": True
                                      }  # 可覆盖输出文件夹
                                )
    print("开始训练模型")
    model.train_model(xtrain, args={'fp16': True})
    result, model_outputs, wrong_predictions = model.eval_model(xvalid, acc=sklearn.metrics.accuracy_score,
                                                                r2=sklearn.metrics.r2_score,
                                                                )
    print(result)

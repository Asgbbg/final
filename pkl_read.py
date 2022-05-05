import joblib

# 读取pkl文件
tf = joblib.load('cache_dir/cached_dev_bert_128_0_2')
print(tf)
# cv = joblib.load('models/cv.pkl')

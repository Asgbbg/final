import joblib

flag = '2022-04-26-13-45-21'
lda = joblib.load('models/' + flag + '/lda_profile.pkl')
tf = joblib.load('models/' + flag + '/tf.pkl')
cv = joblib.load('models/' + flag + '/cv_profile.pkl')
# print
perplexity = lda.perplexity(tf)
print(perplexity)
file1 = open('models/' + flag + '/perplexity.txt', encoding='utf-8', mode='w')
file1.write("lda模型困惑度:" + str(perplexity) + '\n')
file1.write("lda参数:" + str(lda) + '\n')
file1.write("CV参数:" + str(cv) + '\n')
file1.close()

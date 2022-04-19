from time import sleep

from selenium import webdriver

browser = webdriver.Chrome()
from selenium.webdriver.common.by import By

url = 'https://www.taptap.com/user/4393096/reviews'
# url='https://www.baidu.com'
# header={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36'}
# html=requests.get(url,headers=header).text
# sleep(10.0)
# etree_html=etree.HTML(html)
# content=etree_html.xpath('//*[@id="tap"]/div/main/div[2]/div/div/div[3]/div[7]/div/div[9]/div/div/div[1]/div/div[2]/div[3]/a/div/div/div/p')
browser.get(url)
# text=browser.page_source
sleep(10.0)
title = browser.find_element(by=By.XPATH,
                             value='//*[@id="tap"]/div/main/div[2]/div/div/div[3]/div[3]/div/div[3]/div/div/div[1]/div/div[2]/div[1]/div[2]/div[2]/a/div/p/span').text
text = browser.find_element(by=By.XPATH,
                            value='//*[@id="tap"]/div/main/div[2]/div/div/div[3]/div[3]/div/div[3]/div/div/div[1]/div/div[2]/div[3]/a/div/div/div/p').text
tags = browser.find_elements(by=By.XPATH,
                             value='//*[@id="tap"]/div/main/div[2]/div/div/div[3]/div[3]/div/div[3]/div/div/div[1]/div/div[2]/div[1]/div[2]/div[2]/div/div[2]')[
    0].text
texts = browser.find_elements(by=By.CLASS_NAME, value='text-box__content')[1].text
# tag2=browser.find_element(by=By.XPATH,value='//*[@id="tap"]/div/main/div[2]/div/div/div[3]/div[3]/div/div[3]/div/div/div[1]/div/div[2]/div[1]/div[2]/div[2]/div/div[2]/a[2]/div').text

print(title, text, tags, texts)
# print(content)

from time import sleep

from selenium import webdriver

browser = webdriver.Chrome()
from selenium.webdriver.common.by import By

url = 'https://www.taptap.com/app/232118/review'
browser.get(url)
sleep(10.0)
# titles=[]
# tags=[]
# texts=[]
items = []
select_cl = []
# user_name=browser.find_element(by=By.CLASS_NAME,value='user-detail__name-gender').text
# user_id=browser.find_element(by=By.CLASS_NAME,value='user-detail__id-copy').text
# titles = browser.find_elements(by=By.CLASS_NAME,
#                              value='tap-long-text')
# tags= browser.find_elements(by=By.CLASS_NAME,
#                              value='label-tag-group-wrapper')
# texts= browser.find_elements(by=By.CLASS_NAME, value='text-box__content')

text_total = int(browser.find_element(by=By.CLASS_NAME, value='tab-item__text').text)
print(text_total)
sleep(3.0)
while len(items) != text_total:
    print("获取到的items与总数不符")
    print("目前items为" + str(len(items)))
    print("目前text_total为" + str(text_total))
    browser.execute_script("window.scrollBy(0,3500)")
    print("向下滑动3500pix")
    sleep(3.0)
    print("睡觉3s")
    items = browser.find_elements(by=By.CLASS_NAME, value='review-item')
    if len(items) == 20:
        break
for i in range(text_total):
    try:
        text = items[i].find_element(by=By.CLASS_NAME, value='text-box__content').text
        support = int(items[i].find_element(by=By.CLASS_NAME, value='vote-button--up').text)
        print(text)
        print('\n')
    except Exception as e:
        print("捕捉到异常")
        text = 'null'
        print(text)
    else:
        print("正常加载")

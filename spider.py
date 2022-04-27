from time import sleep
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

catchtext = 1


# user_name=browser.find_element(by=By.CLASS_NAME,value='user-detail__name-gender').text
# user_id=browser.find_element(by=By.CLASS_NAME,value='user-detail__id-copy').text
# titles = browser.find_elements(by=By.CLASS_NAME,
#                              value='tap-long-text')
# texts= browser.find_elements(by=By.CLASS_NAME, value='text-box__content')
def text(id, tags):
    browser = webdriver.Chrome()
    items = []
    data_total = []
    browser.get('https://www.taptap.com/app/' + str(id) + '/review?sort=hot&tag_type=1&tag=' + str(tags))
    sleep(10.0)
    print(str(browser.find_element(by=By.CLASS_NAME, value='tap-long-text__contents').text))
    # text_total = browser.find_element(by=By.CLASS_NAME, value='tab-item__text').text
    # text_total=re.sub('[\u4e00-\u9fa5]','',text_total).strip()
    text_total = 1000
    sleep(3.0)
    if catchtext == 1:
        while len(items) != text_total:
            print("获取到的items与总数不符")
            print("目前items为" + str(len(items)))
            print("目前text_total为" + str(text_total))
            browser.execute_script("window.scrollBy(0,3500)")
            print("向下滑动3500pix")
            sleep(3.0)
            print("睡觉3s")
            items = browser.find_elements(by=By.CLASS_NAME, value='review-item')
            if len(items) > 180:
                break
        for i in range(len(items)):
            try:
                text = items[i].find_element(by=By.CLASS_NAME, value='text-box__content').text
            except Exception as e:
                print("捕捉到异常")
                text = 'null'
            else:
                header = ['tags', 'text']
                data = [tags, text]
                data_total.append(data)
    df = pd.DataFrame(data_total)
    df.to_csv('data/' + tags + '.csv', encoding='utf-8', mode='a')
    browser.close()


if __name__ == '__main__':

    # flag_list = ['操作简单', '福利好']
    flag = '抽卡概率低'
    # 崩坏3:10056
    # 原神:168332
    # 元气骑士：34751
    # 另一个伊甸:148290
    # 坎特伯雷公主:149161
    # 阴阳师:12492
    # 重生细胞:171015
    # 战双帕弥什:130651
    # 深空之眼:213181
    # 明日方舟:70253
    # 和平精英:70056
    # 泰拉瑞亚:194610
    # LOLM:176911
    # 王者荣耀:2301
    # 决战！平安京:61620
    if flag == '体验不错':
        text(213181, flag)
        text(70253, flag)
        text(70056, flag)
        text(194610, flag)
        text(34751, flag)
    if flag == '画面优良':
        text(34751, flag)
        text(213181, flag)
        text(70253, flag)
        text(70056, flag)
        text(130651, flag)
    if flag == '剧情丰富':
        text(130651, flag)
        text(70253, flag)
        text(148290, flag)
        text(149161, flag)
        text(12492, flag)
    if flag == '操作简单':
        text(70056, flag)
        text(213181, flag)
        text(130651, flag)
        text(34751, flag)
        text(10056, flag)
    if flag == '福利好':
        text(148290, flag)
        text(149161, flag)
        text(12492, flag)
        text(130651, flag)
        text(70253, flag)

    if flag == '玩法较差':
        text(213181, flag)
        text(176911, flag)
        text(2301, flag)
        text(168332, flag)
        text(61620, flag)
    if flag == '太肝了':
        text(70253, flag)
        text(168332, flag)
        text(130651, flag)
        text(12492, flag)
        text(149161, flag)
    if flag == '运营不足':
        # text(70056,flag)
        # text(213181,flag)
        # text(130651,flag)
        # text(171015,flag)
        text(12492, flag)
    if flag == '过于氪金':
        # text(70056,flag)
        text(213181, flag)
        text(130651, flag)
        text(12492, flag)
        text(149161, flag)
    if flag == '抽卡概率低':
        text(213181, flag)
        text(130651, flag)
        text(149161, flag)
        text(168332, flag)
        text(10056, flag)

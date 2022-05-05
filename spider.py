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
    browser.get('https://www.taptap.com/app/' + str(id) + '/review?sort=hot&tag_type=-1&tag=' + str(tags))
    sleep(10.0)
    print(str(browser.find_element(by=By.CLASS_NAME, value='tap-long-text__contents').text))
    # text_total = browser.find_element(by=By.CLASS_NAME, value='tab-item__text').text
    # text_total=re.sub('[\u4e00-\u9fa5]','',text_total).strip()
    text_total = 10000
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
            if len(items) > 80:
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
    df.to_csv('data/' + tags + '_add.csv', encoding='utf-8', mode='a')
    # df.to_csv('data/extra.csv',encoding='utf-8',mode='a')
    browser.close()


if __name__ == '__main__':

    # flag_list = ['操作简单', '福利好']
    flag = '运营不足'
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
    # 炉石传说： 213
    # 人类跌落梦境   71417
    # 失落城堡  67396
    # 纪念碑谷2 52276
    # 辐射 避难所    33189
    # 月圆之夜  58885
    # 植物大战僵尸2   54031
    # 武侠乂   152653
    # 光·遇   62448
    # 我想成为创造者   144287
    # 香肠派对  58881
    ##
    #
    #
    if flag == 1:
        text(2301, '体验较差')
        text(168332, '体验较差')
        # text(168332,'画面优良')
        text(168332, '过于氪金')
        text(2301, '过于氪金')
        # text(70253,'画面优良')
        # text(70253,'剧情丰富')

    if flag == '体验不错':
        text(71417, flag)
        text(67396, flag)
        text(52276, flag)
        text(33189, flag)
        text(58881, flag)
    if flag == '画面优良':
        # text(61620, flag)
        # text(67396, flag)
        # text(52276, flag)
        # text(33189, flag)
        text(62448, flag)
    if flag == '剧情丰富':
        text(52276, flag)
        text(58885, flag)
        text(62448, flag)
        text(12547, flag)
        text(55307, flag)
    if flag == '操作简单':
        text(58881, flag)
        #
        # text(213181, flag)
        # text(130651, flag)
        # text(34751, flag)
        # text(10056, flag)
    if flag == '福利好':
        text(58881, flag)
        text(218693, flag)
        text(10056, flag)
        #
        # text(130651, flag)
        # text(70253, flag)

    if flag == '玩法较差':
        text(61620, flag)
        text(152653, flag)
        text(55307, flag)
        text(159465, flag)
        text(139301, flag)
    if flag == '太肝了':
        text(54031, flag)
        text(62448, flag)
        text(55307, flag)
        text(159465, flag)
        text(177635, flag)
    if flag == '运营不足':
        text(213, flag)
        text(71417, flag)
        text(67396, flag)
        text(54031, flag)
        text(62448, flag)
    if flag == '过于氪金':
        text(58881, flag)
        text(213, flag)
        text(54031, flag)
        text(152653, flag)
        text(62448, flag)
    if flag == '抽卡概率低':
        text(55307, flag)
        text(158690, flag)
        text(177635, flag)
        text(168332, flag)
        text(10056, flag)

# Author:Zhang Yuan


'''
ChromeDriver版本：100.0.4896.60
ChromeDriver下载：https://chromedriver.storage.googleapis.com/index.html
注意：Chrome浏览器版本和ChromeDriver版本要保持一致

1.ChromeDriver安装
步骤一：将下载好的chromedriver.exe文件放置到chrome浏览器所在目录:
        C:\Program Files (x86)\Google\Chrome\Application
步骤二：复制该目录配置到Windows系统环境变量中:
        C:\Program Files (x86)\Google\Chrome\Application
步骤三：我的电脑→属性→高级系统设置→环境变量→系统变量→Path→编辑→新建，将复制的目录粘贴确定即可，注意：要一路确定返回。然后重启。

'''


# %%

from selenium import webdriver
import datetime
import time

browser = webdriver.Chrome()
browser.maximize_window()

def login():
    # 打开淘宝登录页，并进行扫码登录
    browser.get("https://www.jd.com")
    time.sleep(3)
    if browser.find_element_by_link_text("你好，请登录"):
        browser.find_element_by_link_text("你好，请登录").click()
        print("======请在30秒内完成登录")
        time.sleep(30)
        browser.get("https://cart.jd.com")
    time.sleep(3)
    now = datetime.datetime.now()
    print('======login success:', now.strftime('%Y-%m-%d %H:%M:%S'))
    time.sleep(5)

def buy(times, choose):
    # 点击购物车里全选按钮
    if choose == 2:
        print("======请手动勾选需要购买的商品")
    while True:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        # 对比时间，时间到的话就点击结算
        if now > times:
            if choose == 1:
                while True:
                    try:
                        if browser.find_element_by_id("J_SelectAll2"):
                            browser.find_element_by_id("J_SelectAll2").click()
                            break
                    except:
                        print("======找不到购买按钮")
            # 点击结算按钮
            while True:
                try:
                    if browser.find_element_by_link_text("去结算"):
                        browser.find_element_by_link_text("去结算").click()
                        print("======结算成功")
                        break
                except:
                    pass

            while True:
                try:
                    if browser.find_element_by_id('order-submit'):
                        browser.find_element_by_id('order-submit').click()
                        now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                        print("======抢购成功时间：%s" % now1)
                except:
                    print("======再次尝试提交订单")
            time.sleep(0.01)

#  定位元素方式三种任何一个都可以使用过，实际使用自由组合。
# （1）id定位 driver.find_element_by_id("id")
# （2）name定位 driver.find_element_by_name("name")
# （3）class定位 driver.find_element_by_class_name("class_name")

#%%
# 抢购主函数
if __name__ == "__main__":
    # 京东是从购物车模式抢购：
    login()
    # times = input("请输入抢购时间，格式如(2018-09-06 11:20:00.000000):")
    # choose = int(input("到时间自动勾选购物车请输入“1”，否则输入“2”："))
    times = "2022-12-14 10:00:00" # 格式如(2018-09-06 11:20:00.000000)
    choose = 2 # "到时间自动勾选购物车请输入“1”，否则输入“2”
    buy(times, choose)





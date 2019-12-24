from selenium import webdriver
import time

DRIVER_PATH="D:\MSC\TA\Exercises\chromedriver.exe"

browser = webdriver.Chrome(executable_path=DRIVER_PATH)
browser.get("https://cmbpde2192.corpnet.ifsworld.com:48080/main/ifsapplications/web/")

""" element = browser.find_element_by_id("lst-ib")
browser.execute_script("arguments[0].removeAttribute('onkeypress');", element)
element.send_keys('I Python')

time.sleep(3)
element.submit() """


#tsf > div.tsf-p > div.jsb > center > input[type="submit"]:nth-child(1)
#lst-ib
#elm = browser.find_element_by_css_selector('#tsf > div.tsf-p > div.jsb > center > input[type="submit"]:nth-child(1)')
#elm.text

username = browser.find_element_by_name('username')
password = browser.find_element_by_name("password")

username.send_keys("alain")
password.send_keys("alain")

time.sleep(3)
logIN = browser.find_element_by_id("submit")
logIN.submit()

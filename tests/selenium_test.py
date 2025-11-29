import time
import base64
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

STREAMLIT_URL = "http://localhost:8501"

def wait_for(driver, selector, timeout=10):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
    )

def wait_for_id(driver, element_id, timeout=10):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, element_id))
    )

def test_navigation():
    driver = webdriver.Chrome()
    driver.get(STREAMLIT_URL)

    wait_for_id(driver, "menu_toggle")

    driver.find_element(By.ID, "menu_toggle").click()

    print("Clicked menu button")

    # time.sleep(1)
    # driver.find_element(By.XPATH, "//button[contains(text(), 'Home')]").click()
    # time.sleep(1)
    # assert "GitNos Predictor" in driver.page_source

    # menu_btn.click()
    # driver.find_element(By.XPATH, "//button[contains(text(), 'Info')]").click()
    # time.sleep(1)
    # assert "Model Information" in driver.page_source

    driver.quit()

# def test_upload_and_predict_no_threshold(tmp_path):
#     driver = webdriver.Chrome()
#     driver.get(STREAMLIT_URL)

#     wait_for(driver, "button[role='button']")
#     driver.find_element(By.CSS_SELECTOR, "button[role='button']").click()
#     time.sleep(1)
#     driver.find_element(By.XPATH, "//button[contains(text(), 'Home')]").click()

#     dummy_img = tmp_path / "test.png"
#     with open(dummy_img, "wb") as f:
#         f.write(base64.b64decode(
#             "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
#         ))

#     file_input = wait_for(driver, "input[type='file']")
#     file_input.send_keys(str(dummy_img))

#     time.sleep(1)
#     driver.find_element(By.XPATH, "//button[contains(text(), 'Run Predictions')]").click()

#     time.sleep(3)
#     assert "Prediction Results" in driver.page_source

#     driver.quit()

# def test_upload_and_predict_with_threshold(tmp_path):
#     driver = webdriver.Chrome()
#     driver.get(STREAMLIT_URL)

#     wait_for(driver, "button[role='button']")
#     driver.find_element(By.CSS_SELECTOR, "button[role='button']").click()
#     time.sleep(1)
#     driver.find_element(By.XPATH, "//button[contains(text(), 'Home')]").click()

#     threshold_input = wait_for(driver, "input[type='text']")
#     threshold_input.send_keys("0.7")

#     dummy_img = tmp_path / "test2.png"
#     with open(dummy_img, "wb") as f:
#         f.write(base64.b64decode(
#             "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
#         ))

#     file_input = wait_for(driver, "input[type='file']")
#     file_input.send_keys(str(dummy_img))

#     time.sleep(1)
#     driver.find_element(By.XPATH, "//button[contains(text(), 'Run Predictions')]").click()

#     time.sleep(3)
#     assert "Threshold Prediction Results" in driver.page_source

#     driver.quit()

# def test_info_health_endpoints():
#     driver = webdriver.Chrome()
#     driver.get(STREAMLIT_URL)

#     wait_for(driver, "button[role='button']")
#     driver.find_element(By.CSS_SELECTOR, "button[role='button']").click()
#     time.sleep(1)
#     driver.find_element(By.XPATH, "//button[contains(text(), 'Info')]").click()

#     time.sleep(1)
#     driver.find_element(By.XPATH, "//button[contains(text(), 'Check Model Health')]").click()
#     time.sleep(2)
#     assert "result" in driver.page_source.lower() or "status" in driver.page_source.lower()

#     driver.find_element(By.XPATH, "//button[contains(text(), 'Get Model Info')]").click()
#     time.sleep(2)
#     assert "result" in driver.page_source.lower()

#     driver.quit()
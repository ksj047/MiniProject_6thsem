import cv2
import numpy as np
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Initialize the Selenium webdriver
driver = webdriver.Chrome()

# Navigate to a webpage
driver.get("https://in.tradingview.com/symbols/NSE-NIFTY/")

while True:
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to isolate hand
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of hand
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours and find the largest one (assuming it's the hand)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Recognize hand gesture
            if y > 300:  # Scroll down
                driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
            elif y < 100:  # Scroll up
                driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_UP)
    
    # Display output
    cv2.imshow('Hand Gesture', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Close the browser
driver.quit()

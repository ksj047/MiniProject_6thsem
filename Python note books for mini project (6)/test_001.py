import cv2
import numpy as np
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Initialize the Selenium webdriver
try:
    driver = webdriver.Chrome()
except Exception as e:
    print(f"Error initializing WebDriver: {e}")
    exit(1)

# Navigate to a webpage
driver.get("https://en.wikipedia.org/wiki/Python_(programming_language)")

while True:
    ret, frame = cap.read()
    
    # Check if frame capture was successful
    if not ret:
        continue
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to isolate hand
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours of hand
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours and find the largest one (assuming it's the hand)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Recognize hand gesture
            try:
                if y > 300:  # Scroll down
                    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
                    time.sleep(0.5)  # Delay for 0.5 seconds
                elif y < 100:  # Scroll up
                    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_UP)
                    time.sleep(0.5)  # Delay for 0.5 seconds
            except Exception as e:
                print(f"Error scrolling: {e}")
    
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

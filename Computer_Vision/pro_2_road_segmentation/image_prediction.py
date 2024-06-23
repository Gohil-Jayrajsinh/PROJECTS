from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best_2.pt')
results = model.predict(source='my_test_1.jpg', conf=0.25)
result_image = results[0].plot()  # This returns an image with bounding boxes drawn on it
result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('result.jpg', result_image_bgr)
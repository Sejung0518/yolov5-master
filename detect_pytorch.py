import torch
#
# # model
# model = torch.hub.load('.', 'custom', path='runs/train/trial/weights/best.pt', source='local')
#
# # image
# img1 = 'data/images/val/1-강원-81-자-1246.jpg'
# img2 = "data/images/val/1-경기-38-거-3906.jpg"
#
# # inference
# model.conf = 0.95
# result1 = model(img1)
# result2 = model(img2)
#
# result1.show()
# # result2.show()
# # result.xyxy[0].print()
# # result.pandas().xyxy[0].save()
# print(result1.pandas().xyxy[0])
# # result1.save()
# print(result1.pandas().xyxy[0].to_json(orient="records"))
# # print(result1.xyxy[0])
# result1.crop(save=True)

import cv2
import numpy as np
from yolov5 import YOLOv5

from yolov5.models import YOLOv5

# Load the pre-trained YOLOv5 model
model = YOLOv5(weights="runs/train/trial/weights/best.pt")

# Load an image for object detection
img = cv2.imread("data/images/val/1-강원-81-자-1246.jpg")

# Run object detection on the image
output = model.detect(img)

# Extract the bounding boxes and class labels from the output
boxes = output[0]["boxes"]
labels = output[0]["labels"]

# Label the objects in the image based on the class labels
for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Save the labeled image
cv2.imwrite("labeled_image.jpg", img)


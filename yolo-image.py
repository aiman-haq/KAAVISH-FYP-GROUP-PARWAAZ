import imp
import numpy as np
import argparse
import time
import cv2
import os


args = {'image': r"Images\car3.jpg", 'yolo': r"yolo-coco",'confidence': 0.6, 'threshold': 0.3}


labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
print(weightsPath)
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []
dic  =  {'bicycle':[],'car': [],'motorcycle':[], 'train':[], 'airplane':[],'bus':[],'truck': [], 'boat':[]}

for output in layerOutputs:
	# loop over each of the detections
	for detection in output:

		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		if confidence > args["confidence"]:
			
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
		
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
		
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)


idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

if len(idxs) > 0:
	for i in idxs.flatten():
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		color = [int(c) for c in COLORS[classIDs[i]]]
		#print(color)
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

		label = LABELS[classIDs[i]]

		if label in dic.keys():
			dic[label].append(confidences[i])
		print(confidences[i], LABELS[classIDs[i]])
yy =22
for key in dic:
	object = dic[key]
	if len(object)!=0:
		text2= str(key) + ":" + str(len(object))
		cv2.putText(image, text2, (6,yy), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
	yy += 8

cv2.imshow("image", image)
cv2.waitKey(0)

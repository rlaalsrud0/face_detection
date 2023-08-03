# USAGE
# python mask_detect_recog.py

# import
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import playsound
from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
from gtts import gTTS

name = ''
# 얼굴 감지 및 마스크를 착용하고 있는지 확인
def detect_and_predict_mask(frame, faceNet, maskNet):
	global name
	# 프레임의 크기를 가져오고 blob 생성
    # blob >> 이미지, 사운드, 비디오와 같은 멀티미디어 데이터를 다룰 때 사용
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# 얼굴 인식 모델에 blob을 입력으로 설정
	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []

	# 얼굴 인식된 영역에서 얼굴 추출, 전처리
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			# 박스 좌표 계산
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# ROI 추출
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				faces.append(face)
				locs.append((startX, startY, endX, endY))
    
	# 하나 이상 얼굴이 감지된 경우에만 예측 수행
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	else:
		time.sleep(3)
		webcam.release()
		cv2.destroyAllWindows()
		name = people_detect()

	return (locs, preds)

# 등록된 사용자인지 인식
def people_detect():
    global name
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print('Could not open Webcam')
        exit()

    while webcam.isOpened():
        status, frame2 = webcam.read()
        if status:
            break
    webcam.release()
    cv2.destroyAllWindows()
    
    model = VGGFace.loadModel()

    backends = ['opencv', 'ssd', 'dlib', 'mtcnn']

    img1_list = []
    for root, dirs, files in os.walk('C:/face_detection/'):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            img1_list.append(file_path)

    for img in img1_list:
        img1 = DeepFace.detectFace(img, detector_backend=backends[3])
        img2 = DeepFace.detectFace(frame2, detector_backend=backends[3])
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        img1_representation = model.predict(img1)[0, :]
        img2_representation = model.predict(img2)[0, :]

        distance_vector = np.square(img1_representation - img2_representation)
        distance = np.sqrt(distance_vector.sum())
        if distance < 0.3:
            print(img.split('/')[2].split('\\')[0])
            name = img.split('/')[2].split('\\')[0]
            print(distance)
        else:
            print('등록된 사용자가 아닙니다.')
            name = 'No registered poeple'
            print(name)
    return name

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

print("[INFO] starting video stream...")

mask_count = 0
name = people_detect()
print(name)

webcam = cv2.VideoCapture(0)

# 실시간 웹캠
while True:
	if not webcam.isOpened():
		webcam = cv2.VideoCapture(0)
	_, frame = webcam.read()
	frame = imutils.resize(frame, width=400)
	# 얼굴 감지 및 마스크를 착용하고 있는지 확인
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# 얼굴 경계 사각형 그리기
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		print(box)
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		# No Mask 카운트가 10번일 경우 마스크를 착용하세요 음성 출력
		if label == 'No Mask':
			mask_count += 1
			print(mask_count)
		else:
			mask_count = 0
		if mask_count == 10:
			playsound.playsound('voice3.mp3')
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		
		label = "{} - {}: {:.2f}%".format(name, label, max(mask, withoutMask) * 100)
		print(name)
		print(label)

		# 레이블 및 경계 사각형 표시
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

webcam.release()
cv2.destroyAllWindows()
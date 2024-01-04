import mediapipe as mp
import pandas as pd
import numpy
import keras
import cv2
from cProfile import label
import threading

from pygame import mixer

cap = cv2.VideoCapture(1)
width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc=cv2.VideoWriter_fourcc(*'DIVX')
out=cv2.VideoWriter('output1.avi',fourcc,60.0,(int(width),int(height)))
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = keras.models.load_model("capstoneV4.h5")
lm_list = []

def make_landmark(results): 
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = frame.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_class(label, img):
    font = cv2.FONT_HERSHEY_PLAIN
    WhereText = (10,30)
    fontScale = 1
    if label == "danger":
        fontColor = (0,0,255)
    else:
        fontColor = (0,255,0)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label),
                WhereText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = numpy.array(lm_list)
    lm_list = numpy.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    print(result)
    if result[0][0] >0.95:
        label = "danger"
    else:
        label = "safe"
    return str(label)

i = 0
warm_up_frames = 60
cnt=0
mixer.init() 
sound=mixer.Sound("alert.wav")

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    i=i+1
    if i > warm_up_frames:
        print("Start..")
        if results.pose_landmarks:
            lm = make_landmark(results)
            lm_list.append(lm)
            if len(lm_list) == 20:
                t1 = threading.Thread(target=detect, args=(model, lm_list, ))
                t1.start()
                lm_list = []
            x_coord = list()
            y_coord = list()
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_coord.append(cx)
                y_coord.append(cy)
            if label == "safe":
                cv2.rectangle(img=frame,
                                pt1=(min(x_coord), max(y_coord)),
                                pt2=(max(x_coord), min(y_coord)-25),
                                color=(0,255,0),
                                thickness=1)
                cnt=0
            elif label == "danger":
                cv2.rectangle(img=frame,
                                pt1=(min(x_coord), max(y_coord)),
                                pt2=(max(x_coord), min(y_coord)-25),
                                color=(0,0,255),
                                thickness=4)
                cnt=cnt+1

            frame = draw_landmark(mpDraw, results, frame)
        frame = draw_class(label, frame)
        if(cnt >100):
            sound.play()
        cv2.imshow("image", frame)
        out.write(frame)
        if cv2.waitKey(1) == ord('s'):
            break

cap.release()
cv2.destroyAllWindows()
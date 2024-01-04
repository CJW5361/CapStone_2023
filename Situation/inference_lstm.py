from cProfile import label
import cv2
import mediapipe as mp
import pandas as pd
import numpy
import keras
import threading
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pygame import mixer


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = keras.models.load_model("capstoneV2.h5")

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
    font = cv2.FONT_HERSHEY_SIMPLEX
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

    if result[0][0] > 0.95:
        label = "danger"
    else:
        label = "safe"
    return str(label)

def image_callback(msg):
    global label, lm_list
    
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

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
        mixer.init() 
        sound=mixer.Sound("alert.wav")
       
        if label == "safe":
            cv2.rectangle(img=frame,
                          pt1=(min(x_coord), max(y_coord)),
                          pt2=(max(x_coord), min(y_coord)-25),
                          color=(0, 255, 0),
                          thickness=1)
            cnt=0
        elif label == "danger":
            cv2.rectangle(img=frame,
                          pt1=(min(x_coord), max(y_coord)),
                          pt2=(max(x_coord), min(y_coord)-25),
                          color=(0, 0, 255),
                          thickness=4)
            cnt=cnt+1
            
        if cnt >100:
            sound.play()
            

        frame = draw_landmark(mpDraw, results, frame)
    frame = draw_class(label, frame)
    cv2.imshow("image", frame)
    cv2.waitKey(1)
    

def main():
    rospy.init_node('perception', anonymous=True)
    image_subscriber = rospy.Subscriber("/camera",Image,image_callback)
    rospy.spin()

if __name__ == '__main__':
    global cnt
    cnt=0
    main()

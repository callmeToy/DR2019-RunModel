#!/usr/bin/python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
Topic_Image ='_image/compressed'
sub_Image = rospy.Subscriber(Topic_Image, CompressedImage,get_image)

def get_image(data):
    Array_JPG = np.fromstring(data, np.uint8)
    cv_image = cv2.imdecode(Array_JPG, cv2.IMREAD_COLOR)
    frame = cv_image
    cv2.imshow('adwa',frame)
    cv2.waitKey(1)
    

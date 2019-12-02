#!/usr/bin/env python3
#---Import---#
#---ROS
import rospy
from std_msgs.msg import Float32, String, Int32MultiArray
from sensor_msgs.msg import CompressedImage
#from sensor_msgs import Image
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge, CvBridgeError
import time
import cv2
import numpy as np
import sys
#from turbojpeg import TurboJPEG

cap1 = cv2.VideoWriter('/home/toy/Downloads/outpy1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (320, 240))
cap2 = cv2.VideoWriter('/home/toy/Downloads/outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (320, 240))
def callback(data):
    
    global bridge
    global pub_Speed
    global pub_Angle
    global jpeg
    global cap1
    try:
        #bgr_array = jpeg.decode(data.data)
        Array_JPG = np.fromstring(data.data, np.uint8)        
        cv_image = cv2.imdecode(Array_JPG, cv2.IMREAD_COLOR)
        cv2.imshow('rgb image', cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
        # cap1.write(cv_image)
        # rospy.loginfo(cv_image)
        # rospy.loginfo(cv_image.shape)
        # rospy.loginfo(sys.version)        
    except CvBridgeError as e:
        rospy.loginfo('AAAA', e)
    # rospy.loginfo('22222222222222')

    # cv2.waitKey(3)

def callback_depth(data):
    global cap2
    global bridge
    global pub_Speed
    global pub_Angle
    global jpeg
    # rospy.loginfo(data.data)
    try:
        #bgr_array = jpeg.decode(data.data)
        Array_JPG = np.fromstring(data.data, np.uint8)
        cv_image = cv2.imdecode(Array_JPG, cv2.IMREAD_COLOR)
        cap2.write(cv_image)
        # rospy.loginfo(cv_image.shape)
        # rospy.loginfo(sys.version)        
    except CvBridgeError as e:
        rospy.loginfo('AAAA', e)
    # rospy.loginfo('AAAAAAAAAA1')
    # cv2.imshow("depth window", cv_image)
    # cv2.waitKey(4)

def talker():
    global pub_Speed
    global pub_Angle
    i = 0
    while not rospy.is_shutdown():
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        i = i+1
        rospy.loginfo(float(i))
        pub_Speed.publish(float(i))
        pub_Angle.publish(float(0))
        img = cv2.putText(img, str(i), (250,300), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('image', img)
        if cv2.waitKey(500) == 27:
            break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    #pub_Speed = rospy.Publisher('Team1_speed', Float32, queue_size=10)
    # pub_Angle = rospy.Publisher('Team1_steerAngle', Float32, queue_size=10)
    bridge = CvBridge()
    # cv2.namedWindow('depth window')
    # cv2.namedWindow('Image window')
    # cv2.startWindowThread()
    #jpeg = TurboJPEG()
    rospy.Subscriber('team1/camera/depth/compressed', CompressedImage, callback_depth)
    rospy.Subscriber('team1/camera/rgb/compressed', CompressedImage, callback)
    
    rospy.init_node('talker', anonymous=True)
    rospy.loginfo('AAAAAAAAAA')
    rospy.spin()
    #talker()    
    try:
        pass
        #talker()
    except rospy.ROSInterruptException as be:   
        rospy.loginfo('ahihi ' + be)
        pass

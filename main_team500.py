#!/usr/bin/python3
#---Import---#
#---ROS
import rospy,sys,os
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage
from BienBaoClasssifier import BienBaoClassifier
import time
import cv2
import numpy as np
import scipy.ndimage as sp
from threading import Thread
import os
try:
	os.chdir(os.path.dirname(__file__))	
	os.system('clear')
	print("\nWait for initial setup, please don't connect anything yet...\n")
	sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except: pass
def print_ros(str_):
    rospy.loginfo(str_)

def nothing(x):
        pass

class Team500_CDS_ROS(object):
    def __init__(self, NameTeam='team500', drawImage=False, drawBird=False, limitSpeed=[0, 60], limitAngle=[-30, 30]):
        self.frame = None
        self.limitSpeed = limitSpeed
        self.limitAngle = limitAngle
        self.Speed = 0
        self.Angle = 0
        self.First_time = True
        self.dsize_cut = [0, 0]
        self.Topic_Image = NameTeam + '_image/compressed'
        self.Topic_Speed = NameTeam + '_speed'
        self.Topic_Angle = NameTeam + '_steerAngle'
        self.pub_Speed = None
        self.pub_Angle = None
        self.sub_Image = None
        self.Speed = 50
        self.drawBird = drawBird
        self.drawImage = drawImage
        self.draw_Top = 80
        self.draw_Bot = 77
        self.BienBao_delay_set = 10
        self.BienBao_flag = 2
        self.BienBao_delay = 0
        self.Goc = 30
        self.draw_TopCenter = 25
        self.draw_BotCenter = 160
        self.BirdA = []
        self.Label_bienbao = ['Re Trai', 'Re Phai']
        self.On_Play = False
        self.Continue = True
        self.Trai = None
        self.Phai = None
        self.Thang = None
        self.Thang_flag = 0
        self.Control = 0
        self.draw_Lane()
        self.run()

    def draw_Lane(self):
        h, w = [240, 320]
        Trai = np.array( [[[0, h//4],[w//2,h//4],[(w//4)*3,h//2],[(w//4)*3,h],[0,h]]], dtype=np.int32)
        self.Trai = cv2.fillPoly( np.zeros([240,320],dtype=np.uint8), Trai, 255 )
        Phai = np.array( [[[w, h//4],[w//2,h//4],[w//4,h//2],[w//4,h],[w,h]]], dtype=np.int32)
        self.Phai = cv2.fillPoly( np.zeros([240,320],dtype=np.uint8), Phai, 255 )
        Thang = np.array( [[[w//4, 0],[(w//4)*3,0],[(w//4)*3,h],[w//4,h]]], dtype=np.int32)
        self.Thang = cv2.fillPoly( np.zeros([240,320],dtype=np.uint8), Thang, 255 )

    def RoadDetect(self, image, Low=[30, 8, 69], High=[66, 35, 99]):
        h, w = image.shape[:2]
        # shadow = self.Shadow(image)
        # LineWhite = self.LineWhite(image)
        # HSV = cv2.cvtColor(image[h - 80:h, w // 2 - 90:w // 2 + 90, :], cv2.COLOR_BGR2HSV)
        # H = HSV[..., 0]
        # S = HSV[..., 1]
        # V = HSV[..., 2]
        Low_HSV = Low
        High_HSV = High
        image = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), np.array(Low_HSV),
                            np.array(High_HSV))
        # print_ros(self.Trai.dtype)
        # print_ros(self.Phai.dtype)
        # print_ros(self.BienBao_flag)
        if self.BienBao_flag == 0:
            image = cv2.bitwise_or(image, image, mask=self.Trai)
        elif self.BienBao_flag == 1:
            image = cv2.bitwise_or(image, image, mask=self.Phai)
        image, pts, K = self.Road_Find(image)
        return image, pts, K

    def Road_Find(self, img, winsize=9, margin=150, minpix=1500):
        h, w = img.shape[:2]
        histogram = np.sum(img[2 * (h // 3):, w // 2 - 30: w // 2 + 30], axis=0)
        F_img = cv2.merge((img, img, img))
        mid_Road = int(np.mean(np.where(histogram == np.max(histogram)))) + w // 2 - 30

        win_heigh = np.int(h / winsize)

        nonzero_y, nonzero_x = img.nonzero()
        F = np.zeros_like(img)
        mid_x = mid_Road
        mid_road = []
        KKKK = 0
        for win in range(winsize):
            win_y_low = h - (win + 1) * win_heigh
            win_y_high = h - win * win_heigh
            win_x_low = mid_x - margin
            win_x_high = mid_x + margin
            cv2.rectangle(F_img, (win_x_high, win_y_high), (win_x_low, win_y_low), (255, 0, 0), 2)

            mid_x_good = ((nonzero_x >= win_x_low) & (nonzero_x < win_x_high)
                          & (nonzero_y >= win_y_low) & (nonzero_y < win_y_high)).nonzero()[0]
            if len(mid_x_good) > minpix:
                mid_x = np.int(np.mean(nonzero_x[mid_x_good]))
                KKKK += 1
            mid_road.append(mid_x_good)
        mid_road = np.concatenate(mid_road)
        #
        mid_x_road, mid_y_road = nonzero_x[mid_road], nonzero_y[mid_road]
        x_fit_plot = np.linspace(0, h - 1, h)
        if len(mid_x_road) > 0:
            mid_fit = np.polyfit(mid_y_road, mid_x_road, 2)
            mid_fit_plot = mid_fit[0] * x_fit_plot ** 2 + mid_fit[1] * x_fit_plot + mid_fit[2]
        if self.drawImage:
            for i, mid in enumerate(mid_fit_plot):
                cv2.circle(F_img, (int(mid), i), 1, (255, 0, 0), -1)
        # print(len(mid_fit_plot))
        return F_img, mid_fit_plot, KKKK

    def BirdEye(self, img):
        h, w = img.shape[:2]
        self.BirdA = [(w // 2 + self.draw_BotCenter, h - self.draw_Bot),
                      (w // 2 - self.draw_BotCenter, h - self.draw_Bot),
                      (w // 2 - self.draw_TopCenter, self.draw_Top),
                      (w // 2 + self.draw_TopCenter, self.draw_Top)]
        src = np.float32(self.BirdA)
        dst = np.float32([[w, h], [0, h], [0, 0], [w, 0]])
        M = cv2.getPerspectiveTransform(src, dst)
        N = cv2.getPerspectiveTransform(dst, src)
        F = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return F, M, N

    def run(self):
        self.pub_Speed = rospy.Publisher(self.Topic_Speed, Float32, queue_size=10)
        self.pub_Angle = rospy.Publisher(self.Topic_Angle, Float32, queue_size=10)
        self.sub_Image = rospy.Subscriber(self.Topic_Image, CompressedImage, self.get_image)
        rospy.init_node('talker', anonymous=True)
        print_ros("Team 500 Let's Go!!!")

        self.On_Play = True
        self.BienBaoClassifier = BienBaoClassifier()
        rospy.spin()

    def Core_thread(self):
        while self.On_Play:
            if self.Continue & (not self.frame is None):
                try:
                    image__ = self.frame.copy()
                    image = self.BirdEye(image__)[0]
                    image, pts, K = self.RoadDetect(image)
                    if self.On_Play:
                        self.Publish_Angle(self.AxisControl(pts, K))
                        self.Publish_Speed(self.Speed)
                    else:
                        self.Publish_Speed(0)
                    cv2.imshow('Team500_DUT ETE', image)
                    if cv2.waitKey(1) == 27:
                        self.On_Play = False
                        print_ros('Turn OFF')
                except BaseException as be:
                    print_ros('{}'.format(be))


    def BienBao_thread(self):
        print_ros('Thread_Online')
        self.Continue = False
        self.BienBaoClassifier = BienBaoClassifier()
        self.Continue = True
        while self.On_Play:
            if not self.frame is None:
                try:
                    self.Goc = 30
                    self.Speed = 50
                    self.BienBao_flag, self.BienBao_VT = self.BienBao(self.frame)
                    if self.BienBao_flag in [0,1]:
                        self.Goc = 45
                        self.Speed = 40
                        self.Thang_flag = False
                        time.sleep(2)
                except BaseException as be:
                    print_ros('{}'.format(be))
                # time.sleep(1)
        print_ros('Thread OFF')

    def BienBao(self, image):
        image_ = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (0, 150, 100), (255, 255, 255))
        image_ = cv2.erode(image_, None, iterations=1)
        image_ = cv2.dilate(image_, None, iterations=2)
        ret, labels = cv2.connectedComponents(image_)
        B = 2
        for r in range(ret):
            y, x = np.where(labels == r)
            if (x.shape[0] > 200) & (image_[y[0], x[0]] != 0):
                image_cut = image[np.min(y):np.max(y), np.min(x):np.max(x),:]
                h, w = image_cut.shape[:2]
                if (0.8 <= h/w <= 1.2) & (0.8 <= w/h <= 1.2):
                    B = self.BienBaoClassifier.detect(image[np.min(y):np.max(y), np.min(x):np.max(x), :])
                    if B in [0, 1]:
                        print_ros('Result: {}'.format(self.Label_bienbao[B]))
                    #     if self.drawImage:
                    #         image = cv2.rectangle(image, (np.min(x), np.min(y)), (np.max(x), np.max(y)), (0, 0, 255), 2)
                    #         image = cv2.putText(image, self.Label_bienbao[B],(np.min(x), np.min(y)),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 255))
                        # self.First_time = False
        return B, ((np.min(x), np.min(y)), (np.max(x), np.max(y)))

    def AxisControl(self, pts, K, top=60, thresh=1):
        # print(pts)
        AxisLine = sp.filters.gaussian_filter1d(pts, 0.5)
        # print(AxisLine)
        if self.BienBao_flag in [0, 1]:
            Control = ((AxisLine[240-top] - 360//2)/180)*self.Goc
            self.Thang_flag = 0
        else:
            if (K < 4) & (self.Thang_flag > 20):
                self.Thang_flag = 0
                Control = self.Control
            else:
                self.Thang_flag += 1
                Control = ((AxisLine[240-top] - 360//2)/180)*self.Goc
        # if self.BienBao_flag in [0, 1] & self.BienBao_delay<=self.BienBao_delay_set:
        #     self.BienBao_delay += 1
        #     if self.Bien
        #     Control = -abs(Control) - 10
        # else:
        #     self.BienBao_delay = 0
        # print(Control)
        if Control > thresh:
            Control -= thresh
            Text = "Turn Right: %d" % Control
        elif Control < -thresh:
            Control += thresh
            Text = "Turn Left: %d" % abs(Control)
        else:
            Control = 0
            Text = "Go On"
        self.Control = Control
        print_ros(Text)
        return Control

    def get_image(self, data):
        try:
            if self.First_time:
                self.First_time=False
                Thread(target=self.BienBao_thread).start()
                Thread(target=self.Core_thread).start()
            self.Continue = True
            Array_JPG = np.fromstring(data.data, np.uint8)
            cv_image = cv2.imdecode(Array_JPG, cv2.IMREAD_COLOR)
            self.frame = cv_image
        except BaseException as be:
            print_ros('{}'.format(be))
            self.Continue = True

    def Publish_Speed(self, speed):
        speed = min(self.limitSpeed[1], speed)
        speed = max(self.limitSpeed[0], speed)
        self.Speed = speed
        self.pub_Speed.publish(float(speed))

    def Publish_Angle(self, angle):
        angle = min(self.limitAngle[1], angle)
        angle = max(self.limitAngle[0], angle)
        self.Angle = angle
        self.pub_Angle.publish(float(angle))


if __name__ == '__main__':
    Team500_CDS_ROS(NameTeam='team500', drawImage=True)

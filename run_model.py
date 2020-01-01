#!/usr/bin/python3
#---Import---#
#---ROS

import os
import gc
import cv2
import json
import time
import math
import numpy as np
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
from threading import Thread
import scipy.ndimage as sp
from sensor_msgs.msg import CompressedImage
from sklearn.model_selection import train_test_split
import rospy,sys,os
import rospkg
import sys

os.system("rm -rf ~/.nv")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import keras.backend as K
from keras.layers import Conv2D, MaxPool2D, Dropout, UpSampling2D, Input
from keras.models import *
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import *

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

pkg_path = rospkg.RosPack().get_path('team504')

try:
	os.chdir(os.path.dirname(__file__))	
	os.system('clear')
	print("\nWait for initial setup, please don't connect anything yet...\n")
	sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except: pass

def ros_print(content):
        rospy.loginfo(content)

class CarControl:

    turning_index = 0
    # Map 1
    turning_timing = [500, 200, 100, 0]
    turning_duration = [2200, 2200, 2000, 1500]

    # Map 2
    # turning_timing = [0, 0, 400, 300]
    # turning_duration = [2200, 2200, 2200, 2000]
    
    # Map 3
    # turning_timing = [0, 100]
    # turning_duration = [1800, 1800]

    # Map 4
    # turning_timing = [300, 0, 0]
    # turning_duration = [1300, 1500, 1000]

    # Map 5
    # turning_timing = [100, 500, 100, 0, 200]
    # turning_duration = [1100, 1300, 1100, 1100, 1200]

    distances = None

    prev_I = 0
    prev_error = 0
    last_itr = 0
    tm = None

    start_turning = 0
    left_turn_count = 0
    right_turn_count = 0
    prepare_to_turn = False
    prev_sign = 0 #0 is none, -1 is left, 1 is right
    last_sign_spotted = 0
    last_sign_lost = 0
    last_turn_finished = 0

    fetching_image = True
    segmenting = False
    cropped_sign = None
    sign = None
    road = None
    car = None
    sign_image = None
    car_image = None
    
    road_error = 0
    sign_error = 0
    turning = False
    final_speed = 0
    final_angle = 0
    
    ready = False

    mean_time = 0
    sum_time = 0
    frame_count = 0
    last_image_time = 0
    base_time = 55.5

    def __init__(self, image_size = (240, 320), speed_limit=50, angle_limit = 50):
        '''
        Car Control class init function
        '''
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.constrain_angle = angle_limit
        self.constrain_speed = speed_limit

        self.tm = TimeMetrics()
        self.last_itr = self.tm.millis()

        self.deg_5 = np.zeros((240, 320))
        cv2.line(self.deg_5, (180, 0), (160, 240), (1), 2)
        cv2.line(self.deg_5, (140, 0), (160, 240), (1), 2)

        self.deg_15 = np.zeros((240, 320))
        cv2.line(self.deg_15, (224, 0), (160, 240), (1), 2)
        cv2.line(self.deg_15, (95, 0), (160, 240), (1), 2)

        self.deg_25 = np.zeros((240, 320))
        cv2.line(self.deg_25, (271, 0), (160, 240), (1), 2)
        cv2.line(self.deg_25, (48, 0), (160, 240), (1), 2)

        self.deg_35 = np.zeros((240, 320))
        cv2.line(self.deg_35, (328, 0), (160, 240), (1), 2)
        cv2.line(self.deg_35, (-8, 0), (160, 240), (1), 2)

        self.deg_45 = np.zeros((240, 320))
        cv2.line(self.deg_45, (400, 0), (160, 240), (1), 2)
        cv2.line(self.deg_45, (-80, 0), (160, 240), (1), 2)

        self.deg_55 = np.zeros((240, 320))
        cv2.line(self.deg_55, (502, 0), (160, 240), (1), 2)
        cv2.line(self.deg_55, (-182, 0), (160, 240), (1), 2)
        
        Thread(target=self.seg_thread).start()
        Thread(target=self.road_thread).start()
        Thread(target=self.sign_thread).start()
        Thread(target=self.decide_thread).start()

        print('Car control instance initialized with the following parameters: \n   Image size: {}x{}\n   Speed limit: {}\n   Angle limit: {}'.format(self.image_width, self.image_height, self.constrain_speed, self.constrain_angle))
        print('Ready to connect')

    def refresh_image(self, image):
        self.image_feed = image
        self.fetching_image = False

        time = self.tm.millis() - self.last_image_time
        if time < 200:
            self.sum_time += time
            if self.frame_count != 0:
                self.mean_time = self.sum_time / self.frame_count
            self.frame_count += 1
        self.last_image_time = self.tm.millis()

    def save_sign(self, sign):
        path = pkg_path + '/scripts/Signs_data/{}.jpg'.format(self.tm.millis())
        cv2.imwrite(path, sign)
    
    def sign_thread(self):
        sign_model = self.get_run_model(pkg_path + '/scripts/Saved Models/Model/Sign.json', pkg_path + '/scripts/Saved Models/Weights/sign.h5')
        # sign_model = self.get_run_model(pkg_path + '/scripts/Saved Models/Model/Sign_new.json', pkg_path + '/scripts/Saved Models/Weights/sign_new.h5')
        pred_img = np.zeros((1, 64, 64, 3))
        prediction = sign_model.predict(pred_img)
        print('Sign thread online')

        while True:
            if self.sign is not None or (self.left_turn_count > 0 or self.right_turn_count > 0):
                if (self.cropped_sign is not None):
                    # self.save_sign(self.cropped_sign)
                    self.cropped_sign = cv2.cvtColor(self.cropped_sign, cv2.COLOR_BGR2RGB)
                    self.get_turn_direction(sign_model.predict(np.expand_dims(self.cropped_sign, axis=0)))
                    self.cropped_sign = None

                elif (self.left_turn_count > 0 or self.right_turn_count > 0):
                    if (self.tm.millis() - self.last_sign_spotted) > 2100:
                        print('Safety timer triggered')
                        self.prepare_to_turn = False
                        self.right_turn_count = 0
                        self.left_turn_count = 0
                        self.turning_index = 0
                        self.start_turning = 0
                        self.turning = False

                if (not self.prepare_to_turn):
                    if self.right_turn_count > 0 or self.left_turn_count > 0:
                        if self.start_turning == 0: #if the car is not starting to turn
                            self.start_turning = self.tm.millis()
                        elif self.tm.millis() - self.start_turning <= self.turning_timing[self.turning_index] * self.mean_time / self.base_time:
                            pass
                        elif self.tm.millis() - self.start_turning <= self.turning_duration[self.turning_index] * self.mean_time / self.base_time: #if the car is already turning
                            if self.left_turn_count == self.right_turn_count:
                                if self.prev_sign != 0:
                                    self.right_turn_count += self.prev_sign

                            if self.left_turn_count > self.right_turn_count:
                                # print('Turning left')
                                self.turning = True
                                self.final_speed = 30
                                self.final_angle = -30
                                pass
                            elif self.right_turn_count > self.left_turn_count:
                                # print('Turning right')
                                self.turning = True
                                self.final_speed = 30
                                self.final_angle = 30
                                pass
                        else: #if the turn procedure finished
                            print('Finished turning')
                            self.sign_image = None
                            self.turning = False
                            self.start_turning = 0
                            self.left_turn_count = 0
                            self.right_turn_count = 0
                            self.turning_index += 1
                            self.last_turn_finished = self.tm.millis()
                            if self.turning_index == len(self.turning_timing):
                                self.turning_index = 0

                self.sign = None
            else:
                time.sleep(0.000001)

    def road_thread(self):
        print('Road thread online')
        while True:
            if not (self.road is None):
                sl = np.sum(self.road[:, :160])
                sr = np.sum(self.road[:, 160:])

                if sl + sr < 50:
                    print('Reset detected')
                    self.prepare_to_turn = False
                    self.right_turn_count = 0
                    self.left_turn_count = 0
                    self.turning_index = 0
                    self.start_turning = 0
                elif (sl < 50):
                    print('Outline right')
                    self.road_error = 200
                elif (sr < 50):
                    print('Outline left')
                    self.road_error = -200
                else:
                    self.distances = self.distance_matrix(self.road)
                    
                    # for ins in distance_matrix:
                    #     cv2.circle(cln_img, (ins[0], ins[1]), 3, (255, 255, 0), -1)
                    #     cv2.circle(cln_img, (ins[2], ins[3]), 3, (255, 255, 0), -1)

                    self.road_error = self.error_matrix_method(self.distances)
                    # print(self.road_error)
                    self.road = None
            else:
                time.sleep(0.000001)

    def seg_thread(self):
        run_model = self.get_run_model(pkg_path + '/scripts/Saved Models/Model/Unet4c-optimized.json', pkg_path + '/scripts/Saved Models/Weights/unet4c-optimized.h5')
        pred_img = np.zeros((1, 240, 320, 3))
        prediction = run_model.predict(pred_img)
        kernel5 = np.ones((5, 5), np.uint8)
        kernel7 = np.ones((7, 7), np.uint8)
        self.ready = True
        print('Segment thread online')
        print('Ready to connect')

        while True:
            if not self.fetching_image:
                self.segmenting = True
                image_feed = self.image_feed.copy()
                pred_img = np.expand_dims(image_feed, axis = 0)
                prediction = run_model.predict(pred_img)

                pred_road = prediction[0, :, :, 0]

                pred_sign = prediction[0, :, :, 2]
                if (np.sum(pred_sign) > 50):
                    self.sign_image = image_feed.copy()

                pred_car = prediction[0, :, :, 3]
                pred_car = 1 - pred_car
                pred_car = np.clip(pred_car - cv2.bitwise_or(pred_road, pred_sign), 0, 1)

                pred_sign = cv2.morphologyEx(pred_sign, cv2.MORPH_OPEN, kernel5)
                pred_sign = cv2.morphologyEx(pred_sign, cv2.MORPH_CLOSE, kernel5)
                self.sign = pred_sign
                bnds = self.get_bounding_rect(pred_sign)
                
                if bnds is None or len(bnds) == 0:
                    if self.prepare_to_turn:
                        self.prepare_to_turn = False
                        print('Start turning')
                else:
                    for rect in bnds:
                        offset_h = int(rect[3] * 0.2) * 0
                        offset_w = int(rect[2] * 0.2) * 0
                        cropped = self.sign_image[rect[1] - offset_h:rect[1] + rect[3] + offset_h, rect[0] - offset_w:rect[0] + rect[2] + offset_w]
                        self.cropped_sign = cv2.resize(cropped, (64, 64))
                    self.last_sign_spotted = self.tm.millis()
                self.segmenting = False
                pred_car = cv2.morphologyEx(pred_car, cv2.MORPH_OPEN, kernel5)
                pred_car = cv2.morphologyEx(pred_car, cv2.MORPH_CLOSE, kernel5)
                
                pred_road = cv2.morphologyEx(pred_road, cv2.MORPH_OPEN, kernel7)
                pred_road = cv2.morphologyEx(pred_road, cv2.MORPH_CLOSE, kernel7)
                self.road = pred_road

                if len(np.where(pred_car == 1)[0] > 0):
                    self.car_image = np.copy(pred_road)
                
                self.car = pred_car
            else:
                if self.tm.millis() - self.last_image_time > 2 * self.mean_time:
                    self.fetching_image = True
                time.sleep(0.000001)

    def decide_thread(self):
        print('Control thread online')
        while True:
            if (not self.fetching_image) and self.ready:
                if not self.turning:
                    curr_speed = 60
                    offset = 0
                    kP = 0.10
                    kI = 0.005
                    kD = 320

                    car_pxs = len(np.where(self.car == 1)[0])
                    if car_pxs > 500:
                        curr_speed -= 5
                        kP += 0.02
                        kI += 0.0
                        offset = self.find_car_offset(self.car_image, self.car, 4)

                    angle = self.calc_pid(self.road_error, kP, kI, kD)

                    # if self.left_turn_count > 0 or self.right_turn_count > 0:
                    if self.prepare_to_turn:
                        curr_speed -= 15
                        print('Prepare to turn ', end='')
                        if (self.left_turn_count > self.right_turn_count):
                            print('left')
                        elif (self.left_turn_count < self.right_turn_count):
                            print('right')
                        else:
                            print('up the music ey ey ey')

                    self.final_speed = curr_speed - abs(angle * 1.2)
                    self.final_angle = angle + offset
                    self.fetching_image = True
            else:
                time.sleep(0.000001)

    def draw_left_mask(self, road):
        road_trace = np.where(road[:, :50] == 1)
        min_y = road_trace[1].min()
        max_y = road_trace[1].max()
        yy = (road_trace[0][road_trace[1] == min_y].min(), road_trace[0][road_trace[1] == max_y].min())
        xx = (min_y, max_y)
        poly = np.polyfit(xx, yy, 1)
        if poly[0] < -0.5:
            return road
        yy = poly[0] * np.arange(320) + poly[1]
        
        right_trace = np.where(road[120:, 200:] == 0)
        if (len(right_trace[1]) == 0):
            umin_x = 200
        else:
            umin_x = right_trace[0].min() + 200
        
        for i in range(320):
            road[0:int(yy[i]), i] = 0
            if i > umin_x:
                road[:, i] = 0
        return road

    def draw_right_mask(self, road):
        road_trace = np.where(road[:, 270:] == 1)
        min_y = road_trace[1].min()
        max_y = road_trace[1].max()
        yy = (road_trace[0][road_trace[1] == min_y].min(), road_trace[0][road_trace[1] == max_y].min())
        xx = (min_y + 270, max_y + 270)
        poly = np.polyfit(xx, yy, 1)
        if poly[0] > 0.5:
            return road
        yy = poly[0] * np.arange(320) + poly[1]
        
        right_trace = np.where(road[120:, :120] == 0)
        if (len(right_trace[1]) == 0):
            umin_x = 200
        else:
            umin_x = right_trace[0].max()
        
        for i in range(320):
            road[0:int(yy[i]), i] = 0
            if i < umin_x:
                road[:, i] = 0
        return road
        
    def get_run_model(self, model_link, weight_link):
        '''
        Grab the model
        '''
        if model_link is not '':
            with open(model_link, 'r') as model_read:
                run_model = model_from_json(model_read.read())

            print('Model sucessfully loaded, summmary: ')
            # print(run_model.summary())

            run_model.load_weights(weight_link)
            print('Model weights loaded')
            # run_model._make_predict_function()
            self.graph = tf.get_default_graph()
            # K.clear_session()
            return run_model
        else:
            run_model = load_model(weight_link)
            return run_model

    def get_road_center(self, predicted, segment_count = 8):
        '''
        Return segment_count centers of the road [center0, center1, ... center n]
        '''
        predicted = (predicted * 255).astype(np.uint8)
        interval = int(self.image_height / segment_count)
        centers = np.empty((segment_count, 2), np.uint16)
        for i in range(segment_count):
            temp = predicted[i*interval:(i+1)*interval, :]
            M = cv2.moments(temp)
            if (M["m00"] != 0):
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"]) + i*interval
                centers[i] = (cX, cY)
            else:
                centers[i] = (0, 0)
        return centers

    def calc_pid(self, error, kP, kI, kD):
        '''
        Return the calculated angle base on PID controller
        '''
        if self.last_itr == 0:
            self.last_itr = self.tm.millis()
            return 0
        else:
            itr = self.tm.millis() - self.last_itr
            i_error = error + self.prev_I / itr
            d_error = (error - self.prev_error) / itr

            self.last_itr = itr
            self.prev_I = i_error
            self.prev_error = error
            pid_value = kP * error + kI * i_error + kD * d_error

            # print('Raw pid: {}'.format(pid_value))

            pid_value = np.clip(pid_value, -self.constrain_angle, self.constrain_angle)
            return pid_value

    def error_poly_method(self, centers): # Needs review
        centers = centers.astype(np.int16)
        centers = centers[int(len(centers)/2):len(centers)-1]
        mid_fit = np.polyfit(centers[:, 0], centers[:, 1], 2)
        point = centers[int(len(centers) * 0.5), 0]
        point_y = mid_fit[0] * point ** 2 + mid_fit[1] * point + mid_fit[2]
        angle = math.atan(abs(point - centers[-1, 0])/abs(point_y - centers[-1, 1]))
        if (point - centers[-1, 0] < 0):
            angle *= -1
        return angle

    def error_matrix_method(self, matrix):
        matrix = matrix.astype(np.int64)
        # print(matrix)
        weights = [0.1, 0.3, 0.7, 1.1, 1.5, 1.7]
        error = 0
        for i in range(5):
            error += (matrix[i, 1] - matrix[i, 3]) * weights[i]
            # print(error)
        return error * 0.5
        
    def get_bounding_rect(self, sign):

        if np.sum(sign) < 80:
            return None

        sign = (sign*255).astype(np.uint8)
        ret, labels = cv2.connectedComponents(sign)
        lbls = np.unique(labels)
        bnds = []
        area = 80
        for l in reversed(lbls[:ret]):
            on = np.where(labels == l)
            bndb = [on[1].min(), on[0].min(), on[1].max(), on[0].max()]
            bndb[2] = bndb[2] - bndb[0]
            bndb[3] = bndb[3] - bndb[1]
            if bndb[0] != 0 and bndb[1] != 0 and bndb[0] < 280:
                if bndb[2] * bndb[3] > area:
                    bnds = [bndb]
        bnds = np.array(bnds, np.uint16)
        return bnds

    def get_turn_direction(self, predicted):
        if self.tm.millis() - self.last_turn_finished < 1000:
            return
        predicted = predicted[0]
        if abs(predicted[1] - 1) < 0.1:
            self.prepare_to_turn = True
            if (self.prev_sign == 0):    
                self.prev_sign = 1
            self.right_turn_count += 1
            print('Right sign')
        elif abs(predicted[2] - 1) < 0.1:
            self.prepare_to_turn = True
            if self.prev_sign == 0:    
                self.prev_sign = -1
            self.left_turn_count += 1
            print('Left sign')
        else:
            self.prepare_to_turn = True
            print('No idea')
    
    def find_intersection(self, image, deg):
        inters = np.empty((4))
        finish = 0 #0 is found none, 1 is found right, -1 is found left
        intersect = image * deg
        coor = np.array(np.where(intersect == 1))
        for i in range(coor.shape[1] - 1, -1, -1):
            # print(coor[0, i], coor[1, i])
            if finish == 0:
                if coor[1, i] < self.image_width/2:
                    inters[0] = coor[1, i]
                    inters[1] = coor[0, i]
                    finish = -1
                elif coor[1, i] >= self.image_width/2:
                    inters[2] = coor[1, i]
                    inters[3] = coor[0, i]
                    finish = 1
            elif finish == 1:
                if coor[1, i] < self.image_width/2:
                    inters[0] = coor[1, i]
                    inters[1] = coor[0, i]
                    break
            elif finish == -1:
                if coor[1, i] >= self.image_width/2:
                    inters[2] = coor[1, i]
                    inters[3] = coor[0, i]
                    break
        
        return inters
                    
    def distance_matrix(self, road):
        intersections = np.empty((5, 4), np.uint16)
        road = (road * 255).astype(np.uint8)
        ctns, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        zeros = np.zeros(road.shape)
        zeros = cv2.drawContours(zeros, ctns, -1, (1), 2)
        zeros[220:, :] = 0
        
        intersections[0] = self.find_intersection(zeros, self.deg_5)
        intersections[1] = self.find_intersection(zeros, self.deg_15)
        intersections[2] = self.find_intersection(zeros, self.deg_25)
        intersections[3] = self.find_intersection(zeros, self.deg_35)
        intersections[4] = self.find_intersection(zeros, self.deg_45)
        
        return intersections
    
    def find_car_offset(self, road, car, offset = 6):
        car_center = None
        M = cv2.moments(car)
        if (M["m00"] != 0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            car_center = (cX, cY)
        else:
            print('Cannot find car center')
            return 0

        road_center = None
        pxs = np.where(car == 1)
        road[:pxs[0][0]] = 0
        road[pxs[0][-1]:] = 0
        M = cv2.moments(road)
        if (M["m00"] != 0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            road_center = (cX, cY)
        else:
            print('Cannot find road center')
            return 0

        if abs(car_center[0] - road_center[0]) > 100:
            return 0
        elif car_center[0] > road_center[0]:
            print('Car on the right')
            return -offset
        elif car_center[0] < road_center[0]:
            print('Car on the left')
            return offset
        else:
            print('What the fuk')
            return 0
    
    def get_next_control(self): 
        '''
        Return [speed, streering angle] of the next control
        '''
        # print('Speed: {0:.0f}'.format(self.final_speed), end='')
        # print(', Angle: {0:.2f}'.format(self.final_angle))

        # print('Mean image time: {}'.format(self.mean_time))
        return [self.final_speed, self.final_angle]

class TimeMetrics:
    def __init__(self):
        pass

    def millis(self):
        return int(round(time.time() * 1000))

class ROSControl:
    pubSpeed = None
    pubAngle = None
    subImage = None

    current_speed = 0
    current_angle = 0
    newControl = False

    newImage = False

    model_link = ''
    weight_link = ''

    tm = None

    def refresh_image(self, data):
        '''
        Callback function to refresh the image feed when there is one available
        '''
        try:
            # if self.cControl.fetching_image:
            if True:
                Array_JPG = np.fromstring(data.data, np.uint8)
                cv_image = cv2.imdecode(Array_JPG, cv2.IMREAD_COLOR)
                self.cControl.refresh_image(cv_image)
                self.newImage = True
        except BaseException as be:
            ros_print('{}'.format(be))
            self.Continue = True

    def __init__(self, teamName):
        '''
        ROSPY init function
        '''
        self.subImage = rospy.Subscriber(teamName + '/camera/rgb/compressed', CompressedImage, self.refresh_image)
        self.pubSpeed = rospy.Publisher(teamName + '/set_speed', Float32, queue_size=10)
        self.pubAngle = rospy.Publisher(teamName + '/set_angle', Float32, queue_size=10)
        rospy.init_node('talker', anonymous=True)
        Thread(target=self.drive_thread).start()
        Thread(target=self.publish_thread).start()
        self.tm = TimeMetrics()
        rospy.spin()

    def publish_thread(self):
        while True:
            if self.newControl:
                self.pubSpeed.publish(self.current_speed)
                self.pubAngle.publish(self.current_angle)
                self.newControl = False
            else:
                time.sleep(0.000001)

    def drive_thread(self):
        '''
        Thread for driving the car
        '''
        print('Drive thread online')
        self.cControl = CarControl()
        while True:
            if self.newImage:
                controls = self.cControl.get_next_control()
                self.current_speed = float(controls[0])
                self.current_angle = float(controls[1])
                self.newImage = False
                self.newControl = True

                if self.cControl.cropped_sign is not None:
                    cv2.imshow('sign', self.cControl.cropped_sign)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # if self.cControl.road is not None:
                #     cv2.imshow('feed', self.cControl.road)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
            else:
                time.sleep(0.1)

if __name__ == '__main__':
    # print(sys.version)
    rosControl = ROSControl('team504')

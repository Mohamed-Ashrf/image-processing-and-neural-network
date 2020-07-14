import cv2
import numpy as np
import matplotlib as plt
import math
import matplotlib.pyplot as plt
import serial
import time


class visualization():
    def __init__(self):
        self.loop()
    def draw_circiles(self,img,cent):
        image_center = (400,400)
        cv2.line(img, (0,400), (800,400), (255, 255, 255), 1)
        cv2.line(img, (400, 0), (400, 800), (255, 255, 255), 1)
        for i in range(50,450,25):
            cv2.circle(img, image_center, i, (255, 255, 255), 1)
            img = cv2.putText(img, str(i), (400,400+i), cv2.FONT_HERSHEY_PLAIN  , 1, (255, 255, 255), 1)
        return img

    def get_data(self):
        data = ser.readline()
        data = data.decode('utf-8')
        print(data[1:])
        if int(data[0]) == 1:
            angle = int(data[1:3])
            dist = int(data[3:])*10
        elif int(data[0]) == 2:
            angle = int(data[1:4])
            dist = int(data[4:])
        return [(angle,dist)]

    def get_data_with_noM(self):
        array = []
        for i in range(30, 60, 1):
            array.append((i, 400))

        return array

    def measurment_round(self,objects,img):
        for i in objects:
            angle,dist = i
            line_width = 5
            if angle < 90*math.pi/180 :
                p1 = int(400+dist*math.sin(angle))
                p2 = int(400-dist*math.cos(angle))
                cv2.line(img, (p1, p2), (p1+line_width, p2+2), (0, 0, 255), 2)
            else :
                p1 = int(400 - dist * math.sin(angle))
                p2 = int(400 + dist * math.cos(angle))
                cv2.line(img, (p1, p2), (p1+line_width, p2+2), (0, 0, 255), 2)

        return img

    def loop(self):
        loc2 = [()]
        i = 0
        image = cv2.imread('C:\\Users\\TELE\\Desktop\\dr.habrouk_proj\\others\\carla.jpg')
        image = np.zeros_like(cv2.resize(image, (800, 800)))
        image = self.draw_circiles(image, 800)

        while True:
            loc = self.get_data()
            if loc != loc2:
                i=i+1
                for current_angle, current_Location in loc:
                    pot_array = [[current_angle * math.pi / 180, current_Location]]
                    image = self.measurment_round(pot_array, image)
            loc2 = loc
            if i > 10:
                break
        cv2.imwrite('C:\\Users\\TELE\\Desktop\\dr.habrouk_proj\\others\\img.jpg', image)

        cv2.imshow('image', image)
        cv2.waitKey(0)


try:
    ser =  serial.Serial()
    ser.baudrate = 9600
    ser.port = 'COM3'
    ser.open()
    visualization()
except:
    pass


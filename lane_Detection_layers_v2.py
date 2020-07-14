import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from threading import *
from numba import jit, cuda
import multiprocessing

class detection_algo(Thread):
    def __init__(self, bright, sat):
        self.right_boundry = 645
        self.left_boundry = 625

        self.fault = 0
        self.total = 0
        self.accuracy = 0
        self.accuracy = 0

        self.direction = 0
        self.direction = 0


        self.bright = bright
        self.sat = sat

    def make_coor(self, im, para):
        try:
            slope, inter = para
        except:
            slope, inter = 0.001, 0.001
        y1 = im.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - inter) / slope)
        x2 = int((y2 - inter) / slope)
        return np.array([x1, y1, x2, y2])

    def avg_slope_intercept(self, img, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            para = np.polyfit((x1, x2), (y1, y2),1)
            slope = para[0]
            intercept = para[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        lfit = np.average(left_fit, axis=0)
        rfit = np.average(right_fit, axis=0)
        left_line = self.make_coor(img, lfit)
        right_line = self.make_coor(img, rfit)
        return (np.array([left_line, right_line]), np.array([left_line[2:], right_line[2:]]))

    def canny(self, ima):
        # Calculate median intensity
        median_intensity = np.median(ima)

        # Set thresholds to be one standard deviation above and below median intensity
        lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
        upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

        gray = cv2.cvtColor(ima, cv2.COLOR_RGB2GRAY)
        # blur = cv2.GaussianBlur(gray, (5, 5), 3)
        blur = cv2.bilateralFilter(gray, 3, 25, 25)

        return cv2.Canny(blur, lower_threshold, upper_threshold)

    def region_of_interset(self, im3):
        height = im3.shape[0]

        poly = np.array([[(300, height - 100), (1100, height - 100), (675, 410), (550, 410)]])

        mask = np.zeros_like(im3)
        cv2.fillPoly(mask, poly, (255, 255, 255))

        return cv2.bitwise_and(im3, mask)

    def display(self, im1, li):
        line_img = np.zeros_like(im1)
        li = li.tolist()
        if li is not None:
            for x in li:
                for x1, y1, x2, y2 in x:
                    cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)

        return line_img

    def main_process(self, input_img):
        image = input_img
        lane_image = np.copy(image)
        canny_image = self.canny(lane_image)
        crop = self.region_of_interset(canny_image)
        lines = cv2.HoughLinesP(crop, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=5)
        lined_image = self.display(lane_image, lines)
        combo_image = cv2.addWeighted(lane_image, .8, lined_image, 1, 1)
        the_filtred_lines, target = self.avg_slope_intercept(crop, lines)

        the_filtrd_lines_img = np.zeros_like(lined_image)
        target_point_one, target_point_two = target[0], target[1]

        two_lines_avg_one = (target_point_one[0] + target_point_two[0]) / 2
        two_lines_avg_two = (target_point_one[1] + target_point_two[1]) / 2

        cv2.line(the_filtrd_lines_img, (the_filtred_lines[0][0], the_filtred_lines[0][1]),
                 (the_filtred_lines[0][2], the_filtred_lines[0][3]), (255, 0, 0), 10)
        cv2.line(the_filtrd_lines_img, (the_filtred_lines[1][0], the_filtred_lines[1][1]),
                 (the_filtred_lines[1][2], the_filtred_lines[1][3]), (255, 0, 0), 10)
        return the_filtrd_lines_img, ([two_lines_avg_one, two_lines_avg_two])

    def cont(self, im):
        im_bw = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        (thresh, im_bw) = cv2.threshold(im_bw, 0, 255, 0)
        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return contours, cX, cY

    def frame_detection_without_printing(self, img):

        hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsvImg[..., 1] = hsvImg[..., 1] * self.sat

        hsvImg[..., 2] = hsvImg[..., 2] * self.bright

        frame2 = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
        grayscaled = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        try:

            output, points = self.main_process(frame2)

            # print(points[0],points[1])
            cv2.circle(frame2, (int(points[0]), int(points[1])), 5, (255, 255, 255), thickness=-1, lineType=8, shift=0)

            right_half = output[350:700, 750:1200]
            left_half = output[350:700, 100:550]
            aa = left_half.copy()

            contours, centerXR, centerYR = self.cont(right_half)
            cv2.drawContours(right_half, contours, -1, (0, 255, 0), 3)
            cv2.circle(right_half, (centerXR, centerYR), 5, (255, 255, 255), thickness=-1, lineType=8, shift=0)

            contours, centerXL, centerYL = self.cont(left_half)
            cv2.drawContours(left_half, contours, -1, (0, 255, 0), 3)
            cv2.circle(left_half, (centerXL, centerYL), 5, (255, 255, 255), thickness=-1, lineType=8, shift=0)

            if points[0] in range(0, self.left_boundry):
                self.direction = "left"
            elif points[0] in range(self.right_boundry, frame2.shape[1]):
                self.direction = "right"
            else:
                self.direction = "center"

            qqq = self.region_of_interset(frame2)
            cv2.line(qqq, (self.left_boundry, 0), (self.left_boundry, qqq.shape[0]), (255, 255, 255), 1)
            cv2.line(qqq, (self.right_boundry, 0), (self.right_boundry, qqq.shape[0]), (255, 255, 255), 1)

        except:
            self.direction = "NAN"

        return self.direction

class dec1(Thread):
    def __init__(self,image):
        self.image = image
        Thread.__init__(self)
        self.array = self.dec_make()

    def dec_make(self):
        self.dir_array = np.array([0, 0, 0])
        self.convolutions = [(1.1, 1.1), (1, .9),(5, .3)]
        for conv in self.convolutions:
            kkk = detection_algo(sat=conv[0], bright=conv[1])

            decetion = kkk.frame_detection_without_printing(self.image)

            if decetion == "left":
                self.dir_array[0] += 1
            elif decetion == "center":
                self.dir_array[1] += 1
            elif decetion == "right":
                self.dir_array[2] += 1

        return self.dir_array

class dec2(Thread):
    def __init__(self, image):
        self.image = image
        Thread.__init__(self)


        self.array = self.dec_make(image)

    def dec_make(self,image):
        self.dir_array = np.array([0, 0, 0])
        self.convolutions = [(.5, .5), (1.5, 1.5), (1, 1)]
        for conv in self.convolutions:
            kkk = detection_algo(sat=conv[0], bright=conv[1])

            decetion = kkk.frame_detection_without_printing(image)

            if decetion == "left":
                self.dir_array[0] += 1
            elif decetion == "center":
                self.dir_array[1] += 1
            elif decetion == "right":
                self.dir_array[2] += 1


        return self.dir_array

class dec3(Thread):
    def __init__(self, image):
        self.image = image
        self.array = self.dec_make(image)
        Thread.__init__(self)



    def dec_make(self,image):
        self.dir_array = np.array([0, 0, 0])
        self.convolutions = [(.9, .9), (1.2, .7), (.7, 1.2)]
        for conv in self.convolutions:

            kkk = detection_algo(sat=conv[0], bright=conv[1])
            decetion = kkk.frame_detection_without_printing(image)

            if decetion == "left":
                self.dir_array[0] += 1
            elif decetion == "center":
                self.dir_array[1] += 1
            elif decetion == "right":
                self.dir_array[2] += 1


        return self.dir_array


def main():
    path = 'C:\\Users\\TELE\\Desktop\\road_detection\\road-video-master\\test_3.mp4'
    cap = cv2.VideoCapture(path)
    while cap:
        x = time.time()
        _, image = cap.read()

        c1 = dec1(image=image)
        c2 = dec2(image=image)
        c3 = dec3(image=image)

        c1.start()
        c2.start()
        c3.start()

        c1.join()
        c2.join()
        c3.join()

        dir_array =(c1.array+c2.array+c3.array)/3

        if dir_array[0] + dir_array[1] + dir_array[2] > 1 :
            if np.argmax(dir_array) == 0:
                direction = "left"
            elif np.argmax(dir_array) == 1:
                direction = "center"
            elif np.argmax(dir_array) == 2:
                direction = "right"

        fps = time.time() - x

        cv2.putText(image, "FPS: {} ".format(round(fps,2)), (15, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.putText(image, "conv arrays: {} {} {} ".format(c2.array,c1.array,c3.array), (15, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.putText(image, "sum array: {}  ".format(np.ceil(dir_array)), (15, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.putText(image, "direction: {} ".format(direction), (15, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


        cv2.imshow("ss", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()



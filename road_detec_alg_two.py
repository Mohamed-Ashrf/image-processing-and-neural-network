import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
# from keras.layers import Dense, Dropout, Flatten
# from keras.models import load_model, Sequential
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
#

def make_coor(im,para):
    try:
        slope, inter = para
    except:
        slope, inter = 0.001,0.001
    y1 = im.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-inter)/slope)
    x2 = int((y2-inter)/slope)
    return np.array([x1,y1,x2,y2])

def avg_slope_intercept(img,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        para = np.polyfit((x1,x2),(y1,y2),1)
        slope = para[0]
        intercept = para[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))

    lfit = np.average(left_fit,axis=0)
    rfit = np.average(right_fit,axis=0)
    left_line = make_coor(img,lfit)
    right_line = make_coor(img,rfit)
    return (np.array([left_line,right_line]),np.array([left_line[2:],right_line[2:]]))

def canny(ima,upper,lower):
    # Calculate median intensity
    median_intensity = np.median(ima)

    # Set thresholds to be one standard deviation above and below median intensity
    lower_threshold = int(max(0, (1.0 - 0.5) * median_intensity))
    upper_threshold = int(min(255, (1.0 + 0.5) * median_intensity))

    gray = cv2.cvtColor(ima,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),3)
    blur = cv2.bilateralFilter(gray, 10, 7, 7)
    return cv2.Canny(blur,lower,upper)

def region_of_interset(im3):
    height = im3.shape[0]
    width  =  im3.shape[0]
    intercept = 130
    poly = np.array([[(10,height-intercept),(1150,height-intercept),(290,380), (145,70)]])


    mask = np.zeros_like(im3)
    cv2.fillPoly( mask, poly, (255,255,255) )

    # poly2 = np.array([[(320,height-intercept),(1100,height-intercept),(600,350),(480,350)]])

    # cv2.fillPoly( mask, poly2, (255,255,255) )
    # , (540, 430)

    return cv2.bitwise_and(im3,mask)

def display(im1,li):
    line_img = np.zeros_like(im1)
    li = li.tolist()
    if li is not None:
        for x in li:
            for x1, y1, x2, y2 in x:
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_img

def main_process(input_img):
    image = input_img
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    crop = region_of_interset(canny_image)
    lines = cv2.HoughLinesP( crop, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=5 )
    lined_image = display(lane_image,lines)
    combo_image = cv2.addWeighted( lane_image, .8,lined_image, 1, 1)
    the_filtred_lines,target = avg_slope_intercept(crop,lines)


    the_filtrd_lines_img = np.zeros_like(lined_image)

    target_point_one,target_point_two = target[0],target[1]

    two_lines_avg_one = (target_point_one[0]+target_point_two[0])/2
    two_lines_avg_two = (target_point_one[1]+target_point_two[1])/2


    cv2.line(the_filtrd_lines_img, (the_filtred_lines[0][0], the_filtred_lines[0][1]), (the_filtred_lines[0][2], the_filtred_lines[0][3]), (255, 0, 0), 10)
    cv2.line(the_filtrd_lines_img, (the_filtred_lines[1][0], the_filtred_lines[1][1]), (the_filtred_lines[1][2], the_filtred_lines[1][3]), (255, 0, 0), 10)
    return the_filtrd_lines_img,([two_lines_avg_one,two_lines_avg_two]),crop

def cont(im):
    im_bw = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    (thresh, im_bw) = cv2.threshold(im_bw, 0, 255, 0)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return contours,cX,cY


def mian_process(image):
    can = canny(image, 150, 100)
    cropped_image = can ##region_of_interset(can)
    lines = cv2.HoughLinesP(cropped_image, 8, np.pi / 180, 100, np.array([]), minLineLength=2, maxLineGap=5)
    lined_image = display(image, lines)
    the_filtred_lines, target = avg_slope_intercept(cropped_image, lines)
    the_filtred_lines, target = avg_slope_intercept(cropped_image, lines)
    the_filtrd_lines_img = np.zeros_like(lined_image)
    target_point_one, target_point_two = target[0], target[1]
    two_lines_avg_one = (target_point_one[0] + target_point_two[0]) / 2
    two_lines_avg_two = (target_point_one[1] + target_point_two[1]) / 2
    cv2.line(the_filtrd_lines_img, (the_filtred_lines[0][0], the_filtred_lines[0][1]),
             (the_filtred_lines[0][2], the_filtred_lines[0][3]), (255, 0, 0), 10)
    cv2.line(the_filtrd_lines_img, (the_filtred_lines[1][0], the_filtred_lines[1][1]),
             (the_filtred_lines[1][2], the_filtred_lines[1][3]), (255, 0, 0), 10)

    return the_filtrd_lines_img


image = cv2.imread("C:\\Users\\TELE\\Desktop\\dr.habrouk_proj\\others\\carla.jpg")
image2 = cv2.imread("D:\\udadity simu\\beta_simulator_windows\\IMG\\IMG\\center_2019_10_26_21_32_51_306.jpg")


m,n,_ = image.shape

cv2.imshow("img1",mian_process(image2))
plt.imshow(image2)
plt.show()


cv2.waitKey(0)
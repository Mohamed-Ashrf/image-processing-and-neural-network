import cv2
import numpy as np 
import pickle
import utlis

frameWidth= 640
frameHeight = 480
dispatcher = 50




def initializeTrackbars(intialTracbarVals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],50, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], 100, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], 50, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], 100, nothing)



def valTrackbars():
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")

    src = np.float32([(widthTop/100,heightTop/100), (1-(widthTop/100), heightTop/100),
                      (widthBottom/100, heightBottom/100), (1-(widthBottom/100), heightBottom/100)])
    #src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    return src


def wrapped(image):
	frameWidth= 640
	frameHeight = 480
	c_img = utlis.undistort(img)
	imgThres,imgCanny,imgColor = utlis.thresholding(c_img)
	imgWarp = utlis.perspective_warp(imgColor, dst_size=(frameWidth, frameHeight),src=valTrackbars())
	return imgWarp


def fi_li_dr(arr):
	for k,count in zip(arr,range(0,len(arr))):
		if count+1 < len(arr):
			cv2.line(wrapped_img, k, arr[count+1], (0,0,255), 3)



def operation(wrap, dis):
	discrete_img_array = []
	points_array = []
	left_lines_array = []
	right_lines_array = []

	frameWidth= 640
	frameHeight = 480

	
	counter = 0
	wrapped_out = wrap
	dispatcher = dis


	for k in range(0,wrapped_out.shape[0], dispatcher):
		discrete_img_array.append(wrapped_out[k:k+dispatcher,:])



	for k in discrete_img_array:
		histogram = utlis.get_hist(k)
		midpoint = int(histogram.shape[0] / 2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint
		new_wrapped_out = cv2.cvtColor(k, cv2.COLOR_GRAY2BGR)
		cv2.circle(new_wrapped_out, (leftx_base,k.shape[0]), 20, (0,255,0), 5)	
		cv2.circle(new_wrapped_out, (rightx_base,k.shape[0]), 20, (0,255,0), 5)	
		points_array.append((leftx_base,rightx_base))
		print((leftx_base,rightx_base))

		cv2.circle(wrapped_img, (leftx_base,counter), 5, (0,0,255), 5)
		#cv2.circle(wrapped_img, (rightx_base,counter), 5, (0,0,255), 5)
		left_lines_array.append((leftx_base,counter))
		right_lines_array.append((rightx_base,counter))

		counter+=dispatcher

	return left_lines_array, right_lines_array




def nothing():
	pass



cap = cv2.VideoCapture("project_video.mp4")
sucs, img = cap.read()
img = cv2.resize(img,(frameWidth,frameHeight))

intialTracbarVals = [42,63,14,87] 




initializeTrackbars(intialTracbarVals)

while True:
	
	sucs, img = cap.read()

	

	#img = cv2.imread("image.jpg");
	img = cv2.resize(img,(frameWidth,frameHeight))
	#img = img[:,0:int(img.shape[1]/2)]

	c_img = img.copy()

	wrapped_out = wrapped(c_img)
	wrapped_img = wrapped_out.copy()
	wrapped_img = cv2.cvtColor(wrapped_img, cv2.COLOR_GRAY2BGR)


	left, right = operation(wrapped_out, dispatcher)

	fi_li_dr(left)
	#fi_li_dr(right)

	cv2.imshow("Result", wrapped_img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# cv2.imshow("OG", wrapped_img)
# cv2.waitKey(0)		
	

import cv2
import numpy as np

window_name = "Edge Map"
image = cv2.imread("./photos/test1/P_20160702_133915.jpg", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow( window_name, cv2.WINDOW_NORMAL );

def nothing(x):
	pass

cv2.createTrackbar( "Min Threshold:", window_name, 0, 500, nothing)

while True:
	cv2.imshow(window_name, image)
	v1 = cv2.getTrackbarPos('Min Threshold:', window_name)
	canny = cv2.Canny(image, v1, 240)
	cv2.imshow("Canny edge", canny)

	# check for ESC key
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
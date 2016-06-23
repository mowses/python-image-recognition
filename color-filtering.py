import cv2
import numpy as np

# configurable vars
projectile_pos = (384, 117);
# PLAYER COLOR
pc_hmin, pc_smin, pc_vmin, pc_hmax, pc_smax, pc_vmax = (0, 0, 0, 89, 255, 81) # BLACK

def nothing(x):
	pass

def getPlayerInPosition(contour):
	x, y, w, h = cv2.boundingRect(contour)
	roi = frame[y : y + h, x : x + w]
	#roi_contour = np.full(contour.shape, (x,y), dtype = int)
	#roi_contour = cv2.subtract(contour, roi_contour)

	contour_region = np.zeros(frame.shape[:2], np.uint8)
	cv2.fillPoly(contour_region, pts = [contour], color = (255,255,255))
	contour_region = contour_region[y : y + h, x : x + w]
	
	roi_mask = mask[y : y + h, x : x + w]
	roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	
	# hsv hue sat value
	lower_color = np.array([pc_hmin, pc_smin, pc_vmin])
	upper_color = np.array([pc_hmax, pc_smax, pc_vmax])

	# get mask for player color
	pc_mask = cv2.inRange(roi_hsv, lower_color, upper_color)
	
	# remove roi_mask from pc_mask
	diff = cv2.bitwise_xor(roi_mask, pc_mask, mask = pc_mask)
	# remove anything that is not inside contour_region
	diff_within_contour = cv2.bitwise_and(diff, contour_region, mask = contour_region)
	
	# get the average color inside mask
	mean = cv2.mean(roi, mask = diff_within_contour)
	
	#cv2.imshow('PLAYER COLOR mask', mask)
	#cv2.imshow('PLAYER COLOR roi', roi)
	#cv2.imshow('PLAYER COLOR roi_mask', roi_mask)
	#cv2.imshow('PLAYER COLOR pc_mask', pc_mask)
	#cv2.imshow('PLAYER COLOR diff', diff)
	#cv2.imshow('PLAYER COLOR contour_region', contour_region)
	cv2.imshow('PLAYER COLOR diff_within_contour', diff_within_contour)
	print mean


#cap = cv2.VideoCapture(0)
frame = cv2.imread('futebol_crop.jpg', cv2.IMREAD_COLOR)

# set initial values
hmin, smin, vmin, hmax, smax, vmax = (0, 190, 1, 10, 255, 255) # RED
kernel = np.ones((5,5), np.uint8)

cv2.namedWindow('result')
cv2.createTrackbar('hmin', 'result', hmin, 179, nothing)
cv2.createTrackbar('smin', 'result', smin, 255, nothing)
cv2.createTrackbar('vmin', 'result', vmin, 255, nothing)
cv2.createTrackbar('hmax', 'result', hmax, 179, nothing)
cv2.createTrackbar('smax', 'result', smax, 255, nothing)
cv2.createTrackbar('vmax', 'result', vmax, 255, nothing)

while True:
	#_, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#color = np.uint8([[[0,0,255]]]) # Blue Green Red
	#hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
	#print hsv_color

	# get info from track bar and appy to result
	hmin = cv2.getTrackbarPos('hmin','result')
	smin = cv2.getTrackbarPos('smin','result')
	vmin = cv2.getTrackbarPos('vmin','result')
	hmax = cv2.getTrackbarPos('hmax','result')
	smax = cv2.getTrackbarPos('smax','result')
	vmax = cv2.getTrackbarPos('vmax','result')

	# hsv hue sat value
	lower_color = np.array([hmin, smin, vmin])
	upper_color = np.array([hmax, smax, vmax])

	mask = cv2.inRange(hsv, lower_color, upper_color)
	res = cv2.bitwise_and(frame, frame, mask = mask)

	#erosion = cv2.erode(mask, kernel, iterations = 1)
	dilation = cv2.dilate(mask, kernel, iterations = 1)

	# opening remove false positives from background
	#opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	# Closing is reverse of Opening, Dilation followed by Erosion.
	# It is useful in closing small holes inside the foreground
	# objects, or small black points on the object.
	closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

	_, contourned = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(contourned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# process countours
	for contour in list(contours):
		point_inside = cv2.pointPolygonTest(contour,(projectile_pos), False)
		if point_inside == -1:  # -1 outside countour, 0 on contour, 1 inside
			continue
		
		getPlayerInPosition(contour)

	# draw contours
	#cv2.drawContours(contourned, contours, -1, (128,255,0), 1)

	#cv2.imshow('frame', frame)
	#cv2.imshow('mask', mask)
	#cv2.imshow('dilation', dilation)
	#cv2.imshow('erosion', erosion)
	#cv2.imshow('opening', opening)
	#cv2.imshow('closing', closing)
	#cv2.imshow('contourned', contourned)
	#cv2.imshow('result', res)

	# check for ESC key
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
#cap.release()
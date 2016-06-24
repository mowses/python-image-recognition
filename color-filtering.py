from __future__ import division
import cv2
import numpy as np

# configurable vars
projectile_pos = (384, 117);

# TEAM COLORS #HSV
teams = [
	(0, 190, 1, 10, 255, 255) # RED
]

# PLAYER COLORS #HSV
player_colors = [
	(0, 0, 0, 89, 255, 81), # BLACK
	(0, 190, 190, 9, 255, 255),
	(9, 190, 190, 18, 255, 255),
	(18, 190, 190, 27, 255, 255),
	(27, 190, 190, 36, 255, 255),
	(36, 190, 190, 45, 255, 255),
	(45, 190, 190, 54, 255, 255),
	(54, 190, 190, 63, 255, 255),
	(63, 190, 190, 72, 255, 255),
	(72, 190, 190, 81, 255, 255),
	(81, 190, 190, 90, 255, 255),
	(90, 190, 190, 99, 255, 255),
	(99, 190, 190, 108, 255, 255),
	(108, 190, 190, 117, 255, 255),
	(117, 190, 190, 126, 255, 255),
	(126, 190, 190, 135, 255, 255),
	(135, 190, 190, 144, 255, 255),
	(144, 190, 190, 153, 255, 255),
	(153, 190, 190, 162, 255, 255),
	(162, 190, 190, 171, 255, 255),
	(171, 190, 190, 0, 255, 255)
]

def nothing(x):
	pass

def getPlayerInPosition(contour):
	ret = ()
	x, y, w, h = cv2.boundingRect(contour)
	roi = frame[y : y + h, x : x + w]
	#roi_contour = np.full(contour.shape, (x,y), dtype = int)
	#roi_contour = cv2.subtract(contour, roi_contour)

	contour_region = np.zeros(frame.shape[:2], np.uint8)
	cv2.fillPoly(contour_region, pts = [contour], color = (255,255,255))
	contour_region = contour_region[y : y + h, x : x + w]
	
	roi_mask = mask[y : y + h, x : x + w]
	roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

	# add roi_mask white pixels to contour_region as black area
	cnt_region_roi_mask = cv2.bitwise_xor(roi_mask, contour_region, mask = contour_region)

	# return total of white pixels
	indexes, area_mask_details = np.unique(cnt_region_roi_mask, return_counts = True)
	indexes = np.where(indexes == 255)[0]
	if indexes.size <= 0:  # no white pixels found
		return ret

	max_pixels_coverage = area_mask_details[indexes[0]]
	
	for idx, player_color in enumerate(player_colors):

		# hsv hue sat value
		pc_hmin, pc_smin, pc_vmin, pc_hmax, pc_smax, pc_vmax = player_color
		lower_color = np.array([pc_hmin, pc_smin, pc_vmin])
		upper_color = np.array([pc_hmax, pc_smax, pc_vmax])

		# get mask for player color
		pc_mask = cv2.inRange(roi_hsv, lower_color, upper_color)
		
		# remove anything that is not inside cnt_region_roi_mask white area
		diff = cv2.bitwise_and(pc_mask, cnt_region_roi_mask, mask = cnt_region_roi_mask)

		# count player color coverage
		indexes, pc_area_white_counts = np.unique(diff, return_counts = True)
		indexes = np.where(indexes == 255)[0]
		if indexes.size <= 0:
			continue

		pc_pixels_coverage = pc_area_white_counts[indexes[0]]
		percent = (pc_pixels_coverage / max_pixels_coverage) * 100

		if ret and percent <= ret[1]:
			continue

		ret = (idx, percent, diff)
		
		# get the average color inside mask
		#mean = cv2.mean(roi, mask = diff)
		
		#cv2.imshow('PLAYER COLOR mask', mask)
		#cv2.imshow('PLAYER COLOR roi', roi)
		#cv2.imshow('PLAYER COLOR roi_mask', roi_mask)
		#cv2.imshow('PLAYER COLOR cnt_region_roi_mask', cnt_region_roi_mask)
		#cv2.imshow('PLAYER COLOR pc_mask', pc_mask)
		#cv2.imshow('PLAYER COLOR contour_region', contour_region)
		#cv2.imshow('PLAYER COLOR diff', diff)
		print 'Player', ret[0], 'color pixels coverage:', pc_pixels_coverage, ' from', max_pixels_coverage, '(', percent, '%) color:', player_color

	return ret


#cap = cv2.VideoCapture(0)
frame = cv2.imread('futebol_crop.jpg', cv2.IMREAD_COLOR)

# set initial values
hmin, smin, vmin, hmax, smax, vmax = teams[0]
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
		
		idx, percent, player_mask = getPlayerInPosition(contour)

	# draw contours
	#cv2.drawContours(contourned, contours, -1, (128,255,0), 1)

	#cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	#cv2.imshow('dilation', dilation)
	#cv2.imshow('erosion', erosion)
	#cv2.imshow('opening', opening)
	#cv2.imshow('closing', closing)
	#cv2.imshow('contourned', contourned)
	cv2.imshow('result', res)
	cv2.imshow('player_mask', player_mask)

	# check for ESC key
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
#cap.release()
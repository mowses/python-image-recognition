from __future__ import division
import cv2
import numpy as np

# min TEAM color and min PLAYER color should return an area of at least #value pixels to proced
# in real world app, this could mean filter team color verification based on
# distance. higher numbers mean closest targets once we have more details about the surface thus
# more color area coverage for both team and player colors
min_team_color_pixels_area = 150
min_player_color_pixels_area = 10

# configurable vars X,Y
projectiles_pos = [
	#(648,416),  # for test1/P_20160702_133915.jpg
	#(956,383), # remover

	(103,93),
	(273,93),
	(443,93),
	(613,93),

	(103,263),
	(273,263),
	(443,263),
	(613,263),

	(360,182),
	(100,313),
	(559,210),
	(623,317)
]

# TEAM COLORS #HSV
teams = {
	'red': (-15, 50, 50, 15, 255, 255), # RED
	'yellow': (15, 50, 50, 45, 255, 255), # YELLOW
	'green': (45, 50, 50, 75, 255, 255), # GREEN
	'cyan': (75, 50, 50, 105, 255, 255), # CYAN
	'blue': (105, 50, 50, 135, 255, 255), # BLUE
	'fuchsia': (135, 50, 50, 165, 255, 255) # FUCHSIA
}

# AVAILABLE PLAYER COLORS #HSV
available_player_colors = {
	'red': (-15, 50, 50, 15, 255, 255), # RED
	'yellow': (15, 50, 50, 45, 255, 255), # YELLOW
	'green': (45, 50, 50, 75, 255, 255), # GREEN
	'cyan': (75, 50, 50, 105, 255, 255), # CYAN
	'blue': (105, 50, 50, 135, 255, 255), # BLUE
	'fuchsia': (135, 50, 50, 165, 255, 255) # FUCHSIA
}

def nothing(x):
	pass

def inRange(roi_hsv, colors):
	# hsv hue sat value
	hmin, smin, vmin, hmax, smax, vmax = colors
	lower_color = np.array([hmin, smin, vmin])
	upper_color = np.array([hmax, smax, vmax])

	# get mask color range
	mask = cv2.inRange(roi_hsv, lower_color, upper_color)

	return mask

def getProjectilesThatHitContour(contour):
	# hold projectiles quantity that hit the current contour
	# because many projectiles can hit the same contour
	projectiles_hit_contour = []

	for projectile_pos in projectiles_pos:
		point_inside = cv2.pointPolygonTest(contour,(projectile_pos), False)
	
		if point_inside == -1:  # -1 outside countour, 0 on contour, 1 inside
			continue # project doesnt hit contour

		projectiles_hit_contour.append(projectile_pos)

	return projectiles_hit_contour

def getWhitePixels(mask):
	indexes, area_mask_details = np.unique(mask, return_counts = True)
	indexes = np.where(indexes == 255)[0]

	if not len(indexes):
		return 0

	return area_mask_details[indexes[0]]

def displayProjectilesHitImage():
	cv2.namedWindow('projectiles positions', flags = cv2.WINDOW_NORMAL)
	hithsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	for projectile_pos in projectiles_pos:
		hithsv[projectile_pos[1],projectile_pos[0]] = [0,255,255]

	hithsv = cv2.cvtColor(hithsv, cv2.COLOR_HSV2BGR)
	cv2.imshow('projectiles positions', hithsv)

def getPlayerInPosition(contour):
	ret = None
	x, y, w, h = cv2.boundingRect(contour)
	roi = frame[y : y + h, x : x + w]
	#roi_contour = np.full(contour.shape, (x,y), dtype = int)
	#roi_contour = cv2.subtract(contour, roi_contour)

	contour_region = np.zeros(frame.shape[:2], np.uint8)
	cv2.fillPoly(contour_region, pts = [contour], color = (255,255,255))
	contour_region = contour_region[y : y + h, x : x + w]
	
	roi_mask = mask[y : y + h, x : x + w]
	team_color_coverage = getWhitePixels(roi_mask)
	
	# check if team color have the minimun area pixels coverage
	if min_team_color_pixels_area:
		team_color_area = getWhitePixels(roi_mask)
		if team_color_area < min_team_color_pixels_area:
			print 'TEAM color', team_color_name, 'contour blob has only', team_color_area, 'white area coverage. minimun is', min_team_color_pixels_area
			return ret

	roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

	# add roi_mask white pixels to contour_region as black area
	cnt_region_roi_mask = cv2.bitwise_xor(roi_mask, contour_region, mask = contour_region)

	# return total of white pixels for player color area coverage
	white_coverage = getWhitePixels(cnt_region_roi_mask)
	
	if not white_coverage > 0:  # no white pixels found
		return ret

	for (color_name, player_color) in available_player_colors.items():

		pc_mask = inRange(roi_hsv, player_color)
		
		# remove anything that is not inside cnt_region_roi_mask white area
		diff = cv2.bitwise_and(pc_mask, cnt_region_roi_mask, mask = cnt_region_roi_mask)

		# count player color coverage
		player_color_area_coverage = getWhitePixels(diff)
		
		if not player_color_area_coverage > 0:
			continue # not a single color found for this color (white color in the mask)

		# check for minimun player color area coverage in pixels
		if player_color_area_coverage < min_player_color_pixels_area:
			print 'Player color', color_name, 'has only', player_color_area_coverage, 'white area coverage. minimun is', min_player_color_pixels_area
			continue

		percent = (player_color_area_coverage / white_coverage) * 100

		if ret and percent <= ret['player_color_percent']:
			#print 'HAVE ALREADY A BETTER CANDIDATE', color_name, percent, ret['player_color_percent']
			continue # we have already a better candidate for player color

		ret = {
			'player_color': color_name,
			'player_color_coverage': player_color_area_coverage,
			'player_color_total_area': white_coverage,
			'player_color_percent': percent,
			'team_color_coverage': team_color_coverage,
			'diff': diff
		}
		
		# get the average color inside mask
		#mean = cv2.mean(roi, mask = diff)
		
		#cv2.imshow('PLAYER COLOR mask', mask)
		#cv2.imshow('PLAYER COLOR roi', roi)
		#cv2.imshow('PLAYER COLOR roi_mask', roi_mask)
		#cv2.imshow('PLAYER COLOR cnt_region_roi_mask', cnt_region_roi_mask)
		#cv2.imshow('PLAYER COLOR pc_mask', pc_mask)
		#cv2.imshow('PLAYER COLOR contour_region', contour_region)
		#cv2.imshow('PLAYER COLOR diff', diff)
		#print 'Player', idx, 'color pixels coverage:', player_color_area_coverage, ' from', white_coverage, '(', percent, '%) color:', player_color

	return ret


#cap = cv2.VideoCapture(0)
frame = cv2.imread('./photos/test1.jpg', cv2.IMREAD_COLOR)

kernel = np.ones((5,5), np.uint8)

cv2.namedWindow('result', flags = cv2.WINDOW_NORMAL)
cv2.createTrackbar('hmin', 'result', 0, 179, nothing)
cv2.createTrackbar('smin', 'result', 0, 255, nothing)
cv2.createTrackbar('vmin', 'result', 0, 255, nothing)
cv2.createTrackbar('hmax', 'result', 0, 179, nothing)
cv2.createTrackbar('smax', 'result', 0, 255, nothing)
cv2.createTrackbar('vmax', 'result', 0, 255, nothing)

# create projectiles hit image
displayProjectilesHitImage()


while True:
	#_, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	players_found = []

	print '####################################################'
	
	for (team_color_name, team_color) in teams.items():
		print '================== TEAM COLOR:', team_color_name

		mask = inRange(hsv, team_color)

		#res = cv2.bitwise_and(frame, frame, mask = mask)

		#erosion = cv2.erode(mask, kernel, iterations = 1)
		#dilation = cv2.dilate(mask, kernel, iterations = 1)

		# opening remove false positives from background
		#opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

		# Closing is reverse of Opening, Dilation followed by Erosion.
		# It is useful in closing small holes inside the foreground
		# objects, or small black points on the object.
		closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		
		_, contourned = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(contourned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		cv2.namedWindow('frame', flags = cv2.WINDOW_NORMAL); cv2.imshow('frame', frame)
		cv2.namedWindow('mask for ' + team_color_name, flags = cv2.WINDOW_NORMAL); cv2.imshow('mask for ' + team_color_name, mask)
		cv2.namedWindow('closing for ' + team_color_name, flags = cv2.WINDOW_NORMAL); cv2.imshow('closing for ' + team_color_name, closing)
		cv2.drawContours(contourned, contours, -1, (128,255,0), 1)
		cv2.namedWindow('contourned for ' + team_color_name, flags = cv2.WINDOW_NORMAL); cv2.imshow('contourned for ' + team_color_name, contourned)
		
		# process contours
		for contour in contours:
			projectiles_hit_contour = getProjectilesThatHitContour(contour)

			# check if contour was hit by any projectile
			if not len(projectiles_hit_contour):
				#print 'No projectiles hit this contour
				continue # no projectiles hit contour
			
			player_found = getPlayerInPosition(contour)

			if player_found is None:
				continue

			players_found.append({
				'team': {
					'color': team_color_name,
					'coverage': player_found['team_color_coverage']
				},
				'player': {
					'color': player_found['player_color'],
					'area': player_found['player_color_total_area'],
					'coverage': player_found['player_color_coverage'],
					'percent': player_found['player_color_percent']
				},
				#'contour': contour,
				'projectiles': projectiles_hit_contour
			})
			
			print 'Found player color:', player_found['player_color'], player_found['player_color_percent']
			print 'Hit by', len(projectiles_hit_contour), 'projectile(s)'
			
			# draw contours
			#cv2.drawContours(contourned, contours, -1, (128,255,0), 1)

			
			#cv2.namedWindow('dilation', flags = cv2.WINDOW_NORMAL); cv2.imshow('dilation', dilation)
			#cv2.imshow('erosion', erosion); cv2.namedWindow('erosion', flags = cv2.WINDOW_NORMAL)
			#cv2.namedWindow('opening', flags = cv2.WINDOW_NORMAL); cv2.imshow('opening', opening)
			#cv2.namedWindow('closing', flags = cv2.WINDOW_NORMAL); cv2.imshow('closing', closing)
			#cv2.namedWindow('contourned', flags = cv2.WINDOW_NORMAL); cv2.imshow('contourned', contourned)
			#cv2.imshow('result', res)
			#cv2.namedWindow('player_mask', flags = cv2.WINDOW_NORMAL); cv2.imshow('player_mask', player_mask)

	print '+++++++++++++ RESULT'
	print players_found
	players_found_filtered = []
	
	# now we apply a filter to remove lowest player team area coverage
	# since the script could return more than one found player for the same projectile
	# this happens because inRange could detect a player team color from a player color pattern
	for (pi1, player1) in enumerate(players_found):
		add = True
		for (pi2, player2) in enumerate(players_found):
			if pi1 == pi2:
				continue

			if player1['team']['coverage'] > player2['team']['coverage']:
				continue

			# check to see if any projectile in player1 is inside player2 projectiles
			if not any(True for x in player1['projectiles'] if x in player2['projectiles']):
				continue

			add = False
			break
		
		if add:
			players_found_filtered.append(player1)
				

	print '+++++++++++ FINAL'
	print players_found_filtered

	# check for ESC key
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
#cap.release()
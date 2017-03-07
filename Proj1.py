import numpy as np
import cv2

matrix = np.array([[632.09914295, 0.0, 326.3590625 ],
		   [0.0, 636.95321625, 270.72390718],
 		   [0.0, 0.0, 1.0]])


dist = np.array([ 0.27540634, -1.2389006, 0.00627349, -0.00170199, 0.79555896]).reshape(5, 1)
objpoints = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64).reshape(4, 3, 1)
projpoints = np.array([[0,0,0],[0,0,-1],[0,1,0],[0,1,-1],[1,0,0],[1,0,-1],[1,1,0],[1,1,-1]], dtype=np.float64).reshape(8, 3, 1)

cap = cv2.VideoCapture(0)
while(True):
	ret, frame = cap.read()
	img = frame
	
	a = cv2.waitKey(1) & 0xFF 	
	
	img = cv2.medianBlur(img,5)
	img1 = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
	ret , thresh1 = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
	cv2.imshow('thresh',thresh1)
	image ,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	for i in range(0, len(contours)):
		con = contours[i]
		if cv2.contourArea(con) > 100:
			epsilon = 0.04*cv2.arcLength(con,True)
	    		approx = cv2.approxPolyDP(con,epsilon,True)    		
			if len(approx) == 4 and hierarchy[0][i][2] > 0:
							
				layer2 = hierarchy[0][i][2];
				epsilon2 = 0.04*cv2.arcLength(contours[layer2],True)
	    			approx2 = cv2.approxPolyDP(contours[layer2],epsilon2,True) 
								
							
				if hierarchy[0][layer2][2] > 0 and len(approx2) == 4 and hierarchy[0][hierarchy[0][layer2][2]][2] < 0: 			
					imgpoints = np.array(approx, dtype=np.float64).reshape(4, 2, 1)
					ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, matrix, dist)					
					imgpoints, jacobian = cv2.projectPoints(projpoints, rvec, tvec, matrix, dist)
					
					  
					point1 = (int(imgpoints[0][0][0]), int(imgpoints[0][0][1]))
					point2 = (int(imgpoints[1][0][0]), int(imgpoints[1][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)
					
					point1 = (int(imgpoints[0][0][0]), int(imgpoints[0][0][1]))
					point2 = (int(imgpoints[2][0][0]), int(imgpoints[2][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)

					point1 = (int(imgpoints[0][0][0]), int(imgpoints[0][0][1]))
					point2 = (int(imgpoints[4][0][0]), int(imgpoints[4][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)
					
					point1 = (int(imgpoints[7][0][0]), int(imgpoints[7][0][1]))
					point2 = (int(imgpoints[6][0][0]), int(imgpoints[6][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)

					point1 = (int(imgpoints[7][0][0]), int(imgpoints[7][0][1]))
					point2 = (int(imgpoints[5][0][0]), int(imgpoints[5][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)

					point1 = (int(imgpoints[7][0][0]), int(imgpoints[7][0][1]))
					point2 = (int(imgpoints[3][0][0]), int(imgpoints[3][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)
					
					point1 = (int(imgpoints[3][0][0]), int(imgpoints[3][0][1]))
					point2 = (int(imgpoints[1][0][0]), int(imgpoints[1][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)
					
					point1 = (int(imgpoints[1][0][0]), int(imgpoints[1][0][1]))
					point2 = (int(imgpoints[5][0][0]), int(imgpoints[5][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)

					point1 = (int(imgpoints[4][0][0]), int(imgpoints[4][0][1]))
					point2 = (int(imgpoints[6][0][0]), int(imgpoints[6][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)					
				
					point1 = (int(imgpoints[4][0][0]), int(imgpoints[4][0][1]))
					point2 = (int(imgpoints[5][0][0]), int(imgpoints[5][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)

					point1 = (int(imgpoints[2][0][0]), int(imgpoints[2][0][1]))
					point2 = (int(imgpoints[6][0][0]), int(imgpoints[6][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)

					point1 = (int(imgpoints[2][0][0]), int(imgpoints[2][0][1]))
					point2 = (int(imgpoints[3][0][0]), int(imgpoints[3][0][1]))
					cv2.line(img, point1, point2, (0,0,255), 2)

					dots = hierarchy[0][layer2][2]
					#cv2.drawContours(img, [contours[dots]], -1, (0, 0, 255), 2)				
					j=0				
					while dots >0:
						j=j+1
						#cv2.drawContours(img, [contours[dots]], -1, (0, 0, 255), 2)			
						dots = hierarchy[0][dots][0]
						
					cv2.putText(img,str(j), ((approx[0][0][0] + approx[2][0][0] + 3)/2 , (approx[0][0][1] + approx[2][0][1])/2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)				
					#cv2.drawContours(img, [con], -1, (0, 0, 255), 2)
					#cv2.drawContours(img, [contours[layer2]], -1, (0, 0, 255), 2)
					
				

	cv2.imshow('1', img)
	if a == ord(' '):
		result = img
		cv2.imshow('2', result)
	if a == ord('q'):
	        break



cap.release()
cv2.destroyAllWindows()


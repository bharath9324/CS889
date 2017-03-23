import numpy as np
import cv2
import cv2.aruco as aruco
#from myThread import myThread
#from cube import *








images_arr = ['faces.jpg','car.jpg']

i = 0



# use predefined 6x6 dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# ID 100, 700px
img = aruco.drawMarker(aruco_dict, 100, 700)
#cv2.imshow('marker',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# setup capture
camera = cv2.VideoCapture(0)


camMatrix = np.array([[ 967.14084813, 0.0, 547.48661951],
            [ 0.0, 968.38540386, 399.77711243],
            [ 0.0, 0.0, 1.0 ]])
# distParams = np.array([[0.19975604, -1.1527921, -0.01679071, -0.02060968, 2.69644357]])

distParams = np.array([ 0.02505536, -0.1382031, 0.01193481, 0.00349419, 0.24150026]).reshape(5, 1)


objpoints = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64).reshape(4, 3, 1)
# 3D coordinates for projected axes

#substitute = cv2.VideoCapture('vid.mp4')

ret, frame = camera.read()
# size1 = frame.shape
# InitGL(size1[0],size1[1])
flag_min = 0
# loop
scale = 1
while True:
    ret, frame = camera.read()
    
    
    projpoints = np.array([[0,0,0],[0,0,-1*scale],[0,1*scale,0],[0,1*scale,-1*scale],[1*scale,0,0],[1*scale,0,-1*scale],[1*scale,1*scale,0],[1*scale,1*scale,-1*scale],[0,1.1,0],[1,1.1,0]], dtype=np.float64).reshape(10, 3, 1)
    # ArUco requires grayscale image
    bwimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(bwimg,25,255,cv2.THRESH_BINARY_INV)	
    # default detection parameters
    parameters =  aruco.DetectorParameters_create()

    # ArUco marker detection
    corners, ids, rejectedImgPoints = aruco.detectMarkers(bwimg, aruco_dict, parameters=parameters)

    x1 = [0,0]
    x2 = [1,1]

    if ids is not None:
        for i in range(0, len(ids)):
            if( ids[i] == 180) :
        # draw detected marker frames and IDs
                frame = aruco.drawDetectedMarkers(frame, corners, ids)
        # compute 3D rotational, translational vectors
                rvecs, tvecs = aruco.estimatePoseSingleMarkers(corners, 0.05, camMatrix, distParams)
        # draw the 3D axes on each marker
            #for i in range(0, len(ids)):
                frame = aruco.drawAxis(frame, camMatrix, distParams, rvecs[i], tvecs[i], 0.05)
                mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None, iterations=1)



# convert approx polygon to useable datastructure
                imgpoints = np.array(corners, dtype=np.float64).reshape(4, 2, 1)

    # get the rotational, translational vectors
                ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, camMatrix, distParams)

    # project the 3D axes onto the 2D image
                imgpoints, jacobian = cv2.projectPoints(projpoints, rvec, tvec, camMatrix, distParams)
                #size1 = frame.shape

                #DrawGLScene()



    
                img =frame
                # point = (cx, cy)
                # while(substitute.isOpened()):
                #     ret, substitute_image = substitute.read()





                substitute_image = cv2.imread(images_arr[1])
                if (flag_min == 1):
                    substitute_image = cv2.imread(images_arr[0])

                size = substitute_image.shape

            # Create a vector of source points.
                pts_src = np.array(
                    [
                        [0,0],
                        [size[1] - 1, 0],

                        [size[1] - 1, size[0] -1],
					    [0, size[0] - 1 ]


                        ],dtype=float
                        );



                pts1 = np.array(corners, dtype=np.float32).reshape(4, 2)
                h, status = cv2.findHomography(pts_src,  pts1)

                im_temp = cv2.warpPerspective(substitute_image, h, (frame.shape[1],frame.shape[0]))
                cv2.fillConvexPoly(img, pts1.astype(int), 0, 16)
                img = img + im_temp
                


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

                point1 = (int(imgpoints[8][0][0]), int(imgpoints[8][0][1]))
                point2 = (int(imgpoints[9][0][0]), int(imgpoints[9][0][1]))
                cv2.line(img, point1, point2, (0,0,255), 2)
                
                x1[0] = point1[0]
                x1[1] = point1[1]
                x2[0] = point2[0]
                x2[1] = point2[1]     

                
                frame =img
                mask, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                elements = []
                                
                for c in range(0, len(contours)):
                    cnt = contours[c]
                    # only take big enough contours
                    if cv2.contourArea(cnt) > 100:
                        epsilon = 0.10*cv2.arcLength(cnt,True)
                        approx = cv2.approxPolyDP(cnt,epsilon,True)
                        
                        M = cv2.moments(cnt)
                        cx = (M['m10']/M['m00'])
                        cy = (M['m01']/M['m00'])
                        point = ((cx), (cy))
                        error = (x2[1] - x1[1])*cx/(x2[0] - x1[0]) - cy + x1[1] - (x2[1] - x1[1])*x1[0]/(x2[0] - x1[0])   
                        #cv2.drawContours(frame, [cnt], 0, (0,255,0), 3)
                        
                        if(error < 4 and error > -4 and (  (x1[0] + 0.06*(x2[0] - x1[0])<cx and x2[0] - 0.06*(x2[0] - x1[0]) > cx)) or ((x1[0] + 0.06*(x2[0] - x1[0])>cx and x2[0] - 0.06*(x2[0] - x1[0]) < cx)) ):
                            cv2.drawContours(frame, [cnt], 0, (255,0,0), 3)
                            elements.append(point)
                if len(elements) == 2:
                    xp = elements[0][0] + elements[1][0] - 0.5*(x2[0] - x1[0]) -x1[0]
                    scale = (xp - x1[0])*2/(x2[0] - x1[0]) + 0.3
                     
                    print scale    
                if len(elements) == 1:
                    scale =1

    
    
    cv2.imshow('f',frame)
    cv2.imshow('t',thresh1)
    
    if cv2.waitKey(5) == 27:
        break

# cleanup

camera.release()
#substitute.release()
cv2.destroyAllWindows()


'''   
    cv2.imshow('thr',thresh1)
                
    cv2.imshow('frame', frame)
    
    mask, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in range(0, len(contours)):
                    cnt = contours[c]
        # only take big enough contours
                    if cv2.contourArea(cnt) > 100:
 		    	epsilon = 0.10*cv2.arcLength(cnt,True)
                        approx = cv2.approxPolyDP(cnt,epsilon,True)
                        M = cv2.moments(cnt)
			cv2.drawContours(frame, [cnt], 0, (0,255,0), 3)                        
			if M['m00'] == 0 :
                            continue
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])

    # find contours
                mask, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c in range(0, len(contours)):
                    cnt = contours[c]
        # only take big enough contours
                    if cv2.contourArea(cnt) > 100:
            # heavy approximation
                        epsilon = 0.10*cv2.arcLength(cnt,True)
                        approx = cv2.approxPolyDP(cnt,epsilon,True)
                        M = cv2.moments(cnt)
                        if M['m00'] == 0 :
                            continue
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])

            # only take 4-pointed shapes with child contours
                        if len(approx) == 4 and hierarchy[0][c][2] > 0 :

                            next_child = hierarchy[0][c][2]
                            epsilon2 = 0.01*cv2.arcLength(cnt,True)
                            approx2 = cv2.approxPolyDP(contours[next_child],epsilon2,True)
                # ignoring black border check for simplicity
                            if len(approx2) == 4 :
                # convert approx polygon to useable datastructure
                                imgpoints = np.array(approx, dtype=np.float64).reshape(4, 2, 1)

             

                                point = (cx, cy)

                                img = frame



                                cv2.putText(img, "minus", point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                                flag_min = 1

                                frame = img




                            if len(approx2) == 12 :
                # convert approx polygon to useable datastructure
                                imgpoints = np.array(approx, dtype=np.float64).reshape(4, 2, 1)

                    # get the rotational, translational vectors
                                ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, camMatrix, distParams)

                    # project the 3D axes onto the 2D image
                                imgpoints, jacobian = cv2.projectPoints(projpoints, rvec, tvec, camMatrix, distParams)



                                point = (cx, cy)

                                img = frame

                                cv2.putText(img, "plus", point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


                                frame =img

'''









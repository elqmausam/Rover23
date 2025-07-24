import cv2 
import numpy as np 
from matplotlib import pyplot as plt 


img = cv2.imread('shape4.png') 
cap = cv2.VideoCapture(0)

# converting image into grayscale image 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


# setting threshold of gray image 

_, threshold = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY) 

cv2.imshow("thre",threshold)
# using a findContours() function 

contours, _ = cv2.findContours( 

    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

  

i = 0
t = 0
  
# list for storing names of shapes 

for contour in contours: 

  


    if i == 0: 

        i = 1

        continue

  

    # cv2.approxPloyDP() function to approximate the shape 

    approx = cv2.approxPolyDP( 

        contour, 0.01 * cv2.arcLength(contour, True), True) 

      

    # using drawContours() function 
    print(contour)
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 5) 

    # finding center point of shape 

    M = cv2.moments(contour) 

    if M['m00'] != 0.0: 

        x = int(M['m10']/M['m00']) 

        y = int(M['m01']/M['m00']) 

     
    
    # putting shape name at center of each shape 

    if len(approx) == 2: 
        t+=1

        cv2.putText(img, 'Cone', (x, y), 

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) 

  


if t >= 1:
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower bound and upper bound for orange color
   lower_bound = np.array([255, 236, 225])	 
   upper_bound = np.array([80, 29, 0])

# find the colors within the boundaries
   mask = cv2.inRange(hsv, lower_bound, upper_bound)
   
  
# displaying the image after drawing contours 

   cv2.imshow('shapes', img) 

  

cv2.waitKey(0) 
cv2.destroyAllWindows() 
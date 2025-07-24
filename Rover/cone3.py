import cv2
import numpy as np
def nothing(x):
    
    pass
cap = cv2.VideoCapture(0)

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
t = 0
font = cv2.FONT_HERSHEY_COMPLEX
while (1):
 
  success, frame = cap.read()
  img = frame.copy()
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
  
  l_h = cv2.getTrackbarPos("LH", "Tracking")
  l_s = cv2.getTrackbarPos("LS", "Tracking")
  l_v = cv2.getTrackbarPos("LV", "Tracking")
  u_h = cv2.getTrackbarPos("UH", "Tracking")
  u_s = cv2.getTrackbarPos("US", "Tracking")
  u_v = cv2.getTrackbarPos("UV", "Tracking")
  l_b = np.array([l_h, l_s, l_v])
  u_b = np.array([u_h, u_s, u_v])

  mask = cv2.inRange(hsv, l_b, u_b)
  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.erode(mask, kernel)
 
  #_, contours, _ =cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  img_contours = np.zeros_like(mask)
  #cv2.drawContours(img_contours, contours, -1, (255,255,255), 2)
  for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 5)
    x=approx.ravel()[0]
    y=approx.ravel()[1]
    if len(approx) == 2:
        t+=1
        cv2.putText(frame, "Cone", (x,y), font, 1, (0,0,0))
        lower_bound = np.array([255, 236, 225])	 
        upper_bound = np.array([80, 29, 0])


        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        if mask is not None:
            print("1")


  
  
  cv2.imshow("frame",img)
  cv2.imshow("mask",mask)

  key = cv2.waitKey(1)
  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()

"""  l_b = np.array([97, 50, 113])
  u_b = np.array([130, 203, 255])"""
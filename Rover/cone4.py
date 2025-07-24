import cv2 
import numpy as np
import imutils


def real_time_shape(show):
    # VIDEO CAPTURE
    cap_video = cv2.VideoCapture(0)

    # RUNS FOREVER
    while (1):
        _, frame = cap_video.read()

        # CANNY EDGE DETECTION
        frameG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(frameG, 200, 200)
        thresh = cv2.threshold(frameG, 127, 255, cv2.THRESH_BINARY)[1]
        # CALLING SHAPE DETECTION FUNCTION
        shapes = shapeDetector(thresh, frame.copy())
        if (show):
            # DISPLAY ORIGINAL
            cv2.imshow('Original Image', frame)

            # DISPLAY CANNY OUTPUT
            cv2.imshow('Edges', edges)

            # DISPLAY THRESH OUTPUT
            cv2.imshow('Threshold', thresh)

            # DISPLAY SHAPE OUTPUT
            cv2.imshow('Shapes', shapes)
        cv2.waitKey(5)

    cap_video.release()
    cv2.destroyAllWindows()


def shapeDetector(image, origimage):

    # RESIZING THE IMAGE
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    # SETTING A THRESHOLD TO CONVERT IT TO BLACK AND WHITE

    # FINDING CONTOURS IN THE B/W IMAGE
    contours = cv2.findContours(
        resized.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    for cntour in contours:
        # CALCULATING THE CENTERgray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        shape = detect(cntour)
        M = cv2.moments(cntour)
        if (M["m00"] == 0):
            cX = 0
            cY = 0
        else:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
        cntour = cntour.astype("float")
        cntour *= ratio
        cntour = cntour.astype("int")
        cv2.drawContours(origimage, [cntour], -1, (34, 0, 156), 2)
        cv2.putText(origimage, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 2)
    return (origimage)

# NEEDS TO BE REPLACED BY K MEANS CLUSTERING INSTEAD OF CONTOUR MAPPING


def detect(c):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"

    elif len(approx) == 4:
        (_, _, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    elif len(approx) == 5:
        shape = "pentagon"

    else:
        shape = "circle"

    return shape


if __name__ == "__main__":
    real_time_shape(1)

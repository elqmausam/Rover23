from _future_ import print_function
import cv2
import os
import numpy as np
import array as arr
import moviepy.editor as moviepy
import sys
mainfolder = '/home/niyati/vihaan_rover/src/cv_basics/src/Images'
myFolders = os.listdir(mainfolder)
print(myFolders)

for folder in myFolders:
    path = mainfolder +'/'+folder
    print(path)
    images=[]
    myList = os.listdir(path)
    print(f"total no of image detected {len(myList)}")
    for i in range(len(myList)):
        """dim=(1024,768)
        imagepath = path+'/'+imgN
        print(imagepath)
        image = cv2.imread(imagepath,cv2.IMREAD_COLOR)
        image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
        images.append(image)"""
        images.append(cv2.imread(myList[i])) 
        print(len(images))
        images[i]=cv2.resize(images[i],(0,0),fx=0.4,fy=0.4)

    cv2.imshow('1',images[0])
    stitcher = cv2.Stitcher.create(mode=0)
    
    ret,pano = stitcher.stitch(images)
    print("3")
    if ret != cv2.STITCHER_OK:
        print("0")
        print(ret)
        sys.exit(-1)

    cv2.imshow("show",pano)
    
    print(pano.shape[0])
    print(pano.shape[1])
    print("2")
    height = pano.shape[0]
    width = pano.shape[1]
    pano = cv2.resize(pano,(2100,700),interpolation=cv2.INTER_AREA)
    pano = pano[100:pano.shape[0]-100,300:pano.shape[1]-300]
    

    for alpha in np.arange(0.5,1,0.1)[::-1]:
        overlay = pano.copy()
        output1 = pano.copy()

        cv2.rectangle(overlay,(433,464),(937,526),(191,165,143),-1)
        #cv2.putText(overlay,"hello".format(alpha),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(58,79,122),3)

        cv2.addWeighted(overlay,alpha,output1,1-alpha,0,output1)
        #print("alpha={},beta={}".format(alpha,1-alpha))

    for alpha in np.arange(0.8,1,0.1)[::-1]:
        overlay = output1.copy()
        outputFinal = output1.copy()

        cv2.putText(overlay,"Hello".format(alpha),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(71,38,10),3)

        cv2.addWeighted(overlay,alpha,outputFinal,1-alpha,0,outputFinal)
        #print("alpha={},beta={}".format(alpha,1-alpha))


        if ret==cv2.STITCHER_OK:
            print("panorama generated")
            cv2.imwrite('home/panorama.jpg',outputFinal)
            cv2.imshow(folder,outputFinal)
            cv2.waitKey(10)
        else:
            print("Error during Stitching")

cv2.waitKey(0)
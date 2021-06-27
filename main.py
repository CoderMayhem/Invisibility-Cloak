import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')

while cap.isOpened():
    #take each frame
    ret, frame = cap.read()

    if ret: 
        #how do we convert rgb to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #cv2.imshow("hsv",hsv)
        red = np.float32([[[201, 45, 42]]])
        hsv_red=cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

        #threshold the hsv value for only red color
        l_red = np.array([0,100,100])
        u_red = np.array([10,255,255])

        mask = cv2.inRange(hsv,l_red,u_red)
        #cv2.imshow("mask",mask)

        #all things red
        part1 = cv2.bitwise_and(back,back, mask=mask)
        #cv2.imshow("part1",part1)

        mask= cv2.bitwise_not(mask)
        
        #all things not red
        part2 = cv2.bitwise_and(frame, frame, mask=mask)
        #cv2.imshow("mask",part2)
        kernels = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(back, cv2.MORPH_OPEN, kernels)
        cv2.imshow("cloak", part1+part2)

        if cv2.waitKey(5)==ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cvzone
import cv2
import math
import numpy as np

cap=cv2.VideoCapture('video.mp4')
model=YOLO('best.pt')

classnames=['fire']

alert=''
fire=False
while True:
    ret,frame = cap.read()
    if fire:
        alert="Fire detected!"
    cv2.putText(frame,str(alert),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),3)
    frame=cv2.resize(frame,(640,480))
    framec=frame.copy()
    blur=cv2.GaussianBlur(frame,(15,15),0)
    hsvtype=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    

    lower=[15,0,100]
    upper=[30,255,255]

    lower=np.array(lower,dtype='uint8')
    upper=np.array(upper,dtype='uint8')

    mask=cv2.inRange(hsvtype,lower,upper)
    output=cv2.bitwise_and(framec,hsvtype,mask=mask)
    result=model(output,stream=True)
    for info in result:
        boxes=info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence=math.ceil(confidence*100)
            Class= int(box.cls[0])
            if confidence > 50:
                fire=True
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                #cvzone.putTextRect(frame,f'{classnames[Class]} {confidence}%',[x1+8,y1+100],scale=1.5,thickness=2)
    cv2.imshow("frame",frame)
    cv2.imshow("output",output)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

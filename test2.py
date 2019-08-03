import cv2
# =============================================================================
# import requests
# import numpy as np
# =============================================================================

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth = cv2.CascadeClassifier('haarcascade_mouth.xml')
nose = cv2.CascadeClassifier('haarcascade_nose.xml') 

cap = cv2.VideoCapture(0)

while True:
# =============================================================================
#     img_rep = requests.get("http://192.168.1.3:8080/shot.jpg")
#     img_array = np.array(bytearray(img_rep.content), dtype=np.uint8)
#     img = cv2.imdecode(img_array, -1)
# =============================================================================
    
    ret, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
        m = mouth.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in m:
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (48, 231, 206), 2)
            
        n = nose.detectMultiScale(roi_gray)
        for (nx, ny, nw, nh) in n:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (170, 231, 48), 2)
    
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
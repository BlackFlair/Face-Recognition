import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'wb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()} # Reverse Key and Value

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1, minNeighbors=5)

    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] # roi : Region Of Interest
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray) # conf : confidence
        if conf >=45:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "my-img.png"

        cv2.imwrite(img_item,roi_gray)

        color = (255,0,0) # BGR scale
        stroke = 2 # Thickness of border line
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, stroke)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
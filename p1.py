import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: # This is line video end after then stop program
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # in 1.1 factor and 5 detection 
    faces = face_cascade.detectMultiScale(gray, 1.1, 5) 

    # in this line show face is available then next
    for (x, y, w, h) in faces:
        # in that show green rectangle of face
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 2)
        
        # ROI (Region of Interest) define - x, y, w, h now ready for work 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # (Eyes Detection)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        
        # finding the eyes then show the message of face
        if len(eyes) > 0:
            # show the text of face's ractangle
            cv2.putText(frame, "Eyes Detected", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            
        
        # (Smile Detection)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        
        # after detected smile show the text
        if len(smiles) > 0:
            # show the text of face's ractangle
            cv2.putText(frame, "Smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
    # show to the display
    cv2.imshow("Smart Face Detector", frame)

    # 'q' for quite   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
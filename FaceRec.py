import cv2
def generatedataset(frame, id, imageid):
    cv2.imwrite("dataset/User."+str(id)+"."+str(imageid)+".jpg",frame)

def drawboundry(frame,classifier,scaleFactor,minNeighbors,color,text, clf):
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_frame,scaleFactor,minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)
        id, _ = clf.predict(gray_frame[y:y+h,x:x+w])
        if id == 1:
            cv2.putText(frame,"Afnan",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            print(x,"\t", y, "\t", w, "\t", h)
        if id == 2:
            cv2.putText(frame,"Aheed",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        coords = [x,y,w,h]
    return coords


def drawboundry1(frame,classifier,scaleFactor,minNeighbors,color,text):
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_frame,scaleFactor,minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)
        cv2.putText(frame,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        coords = [x,y,w,h]
    return coords


def recognize(frame, clf, faceCascade):
    color = {"blue":(255,100,50),"red":(0,0,255),"green":(0,255,0)}
    coords = drawboundry(frame, faceCascade, 1.1, 10, color['blue'], "Face", clf)
    return frame


def detect(frame, faceCascade, eyeCascade, imageid):
    color = {"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0)}
    coords = drawboundry1(frame, faceCascade, 1.1, 10, color['blue'], "Face")

    if len(coords)==4:
        roi = frame[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        userid = 1
        generatedataset(roi, userid, imageid)
    return frame


clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

imageid = 0
while True:
    _, frame = video.read()
    #frame = detect(frame, faceCascade, eyeCascade, imageid)
    frame = recognize(frame, clf, faceCascade)

    cv2.imshow("Video", frame)
    imageid += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
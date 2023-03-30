import cv2
import numpy as np
import face_recognition
import os
import datetime

#importing all images from images attendance
path='images attendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# now finding all the encodings of images
def findencodings(images):
    encodelist=[]
    #now we will loop through the images and covert their bg-color to RGB
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        # now appending our encodings to aur encodinglist array
        encodelist.append(encode)

    return encodelist
def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListKnown = findencodings(images)
print('encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facescurframe = face_recognition.face_locations(imgS)
    encodecurframe = face_recognition.face_encodings(imgS, facescurframe)

    for encodeFace,faceLoc in zip(encodecurframe,facescurframe):
         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
         print(faceDis)
         matchIndex = np.argmin(faceDis)

         if matches[matchIndex]:
             name = classNames[matchIndex].upper()
             print(f"Match found: {name}")
             cv2.rectangle(img, (faceLoc[3] * 4, faceLoc[0] * 4), (faceLoc[1] * 4, faceLoc[2] * 4), (0, 255, 0), 2)
             cv2.putText(img, name, (faceLoc[3] * 4, faceLoc[0] * 4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
             markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
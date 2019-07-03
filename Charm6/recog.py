import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


face_list = []
data = np.load("face_data.npy")
# print(data)

X = data[:,1:]
y = data[:,0]

X = X.astype(np.int64)
# print(X)
# print(y)

model = KNeighborsClassifier(10)
model.fit(X, y)



while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)

        areas = []
        for face in faces:
            x, y, w, h = face
            areas.append((w*h, face))

        if len(faces) > 0:
            face = max(areas)[1]
            x, y, w, h = face

            face_img = gray[y:y+h, x:x+w]

            face_img = cv2.resize(face_img, (100, 100))
            face_flatten = face_img.flatten()
            res  = model.predict([face_flatten])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 55, 120), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(res[0]), (x, y), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

            print(res)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# model = KNeighborsClassifier(10)
# model.fit(X_train, y_train)
# print(model.score(X_test,y_test))
# print(data.shape)
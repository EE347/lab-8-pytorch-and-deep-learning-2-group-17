from picamera2 import Picamera2
import cv2
import os
import time

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

train_folder = 'data/train'
test_folder = 'data/test'
os.makedirs(f'{train_folder}/0', exist_ok=True)
os.makedirs(f'{train_folder}/1', exist_ok=True)
os.makedirs(f'{test_folder}/0', exist_ok=True)
os.makedirs(f'{test_folder}/1', exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

num_images = 60
for teammate in [0, 1]:
    count = 0
    while count < num_images:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (64, 64))
            folder = train_folder if count < 50 else test_folder
            file_path = f'{folder}/{teammate}/{count}.jpg'
            cv2.imwrite(file_path, face_resized)
            count += 1
            print(f"Saved image {count} for teammate {teammate}")

            if count >= num_images:
                break

        cv2.imshow('Face', face_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
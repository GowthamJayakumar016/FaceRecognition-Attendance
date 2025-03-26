import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime

# Load known faces
path = 'faces'  # Folder containing known faces
images = []
classNames = []
faceEncodings = []

for file in os.listdir(path):
    img = face_recognition.load_image_file(f'{path}/{file}')
    enc = face_recognition.face_encodings(img)[0]
    images.append(img)
    classNames.append(os.path.splitext(file)[0])
    faceEncodings.append(enc)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Attendance logging
attendance_file = 'attendance.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Time'])

def mark_attendance(name):
    with open(attendance_file, 'r+') as f:
        lines = f.readlines()
        names = [line.split(',')[0] for line in lines]
        if name not in names:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'{name},{now}\n')

while True:
    success, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(faceEncodings, encoding)
        face_distances = face_recognition.face_distance(faceEncodings, encoding)
        best_match = np.argmin(face_distances)
        
        if matches[best_match]:
            name = classNames[best_match]
            mark_attendance(name)
            
            y1, x2, y2, x1 = [i * 4 for i in location]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

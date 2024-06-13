import cv2
import threading
from home.models import emp_data, data
from datetime import datetime
import face_recognition
import time
import os
import numpy as np


# to capture video class
class VideoCamera_exit(object):
    def __init__(self):
        video_path = 'home\emp_imgs\office_cctv.mp4'
        self.video = cv2.VideoCapture(video_path)
        (self.grabbed, self.frame) = self.video.read()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.face_locations = []  # Initialize face_locations
        threading.Thread(target=self.update, args=()).start()
        self.get_all_encodings()

    def get_all_encodings(self):
        self.emp_data = emp_data.objects.all()
        self.imgs = []
        self.emp_name = []
        self.ap_emp = []
        self.imgs_encoding = []
        
        for img in self.emp_data:
            self.imgs.append(img.img)
            self.emp_name.append(img.name)
            self.ap_emp.append(img.name)
            self.img_load = face_recognition.load_image_file(img.img)
            self.img_encoding = face_recognition.face_encodings(self.img_load)[0]
            self.imgs_encoding.append(self.img_encoding)
        
        print(self.imgs)
        print(self.emp_name)
        print(self.imgs_encoding)

        self.not_exit_emp_raws = data.objects.filter(exit_time = '0').values()
        self.not_exit_emp = []
        for x in self.not_exit_emp_raws:
            self.not_exit_emp.append(x['name'])
        print("not_exit_emp")
        print(self.not_exit_emp)

        self.current_date = datetime.now().strftime("%d/%m/%Y")
        self.enterd_emp = data.objects.filter(date=self.current_date).values()
        # self.enterd_emp_name = [item['name'] for item in self.enterd_emp]
        self.enterd_emp_name = [item['name'] for item in self.enterd_emp]
        # self.enterd_emp_id = [item['id'] for item in self.enterd_emp]
        print("enterd_emp_name")
        print(self.enterd_emp_name)


    def __del__(self):
        self.video.release()

    def get_frame(self):

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []



        # Detect face locations
        self.face_locations = face_recognition.face_locations(self.frame)
        self.face_encodings = face_recognition.face_encodings(self.frame, self.face_locations)

        for self.face_encoding, self.face_location in zip(self.face_encodings, self.face_locations):
            self.matches = face_recognition.compare_faces(self.imgs_encoding, self.face_encoding)
            self.name=""
            self.face_distance = face_recognition.face_distance(self.imgs_encoding, self.face_encoding)
            self.best_match_index = np.argmin(self.face_distance)
            if self.matches[self.best_match_index]:
                self.name = self.emp_name[self.best_match_index]
                print(self.name)
                # Draw rectangles around detected faces
                top, right, bottom, left = self.face_location
                cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # font = cv2.FONT_HERSHEY_PLAIN
                # cv2.putText(self.frame, self.name, (left + 6, top - 20), font, 1.0, (255, 255, 255), 1)
            # else:
                # Draw rectangles around detected faces
                # top, right, bottom, left = self.face_location
                # cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
            self.face_names.append(self.name)
            if self.name in self.emp_name:
                if self.name in self.enterd_emp_name:
                    if self.name in self.not_exit_emp:        
                        self.not_exit_emp.remove(self.name)
                        self.enterd_emp_name.remove(self.name)
                        self.current_fream_time = datetime.now().strftime("%H:%M:%S")
                        self.current_fream_date = datetime.now().strftime("%d/%m/%Y")
                        self.update_quary = data.objects.get(name=self.name, exit_time='0')
                        self.update_quary.exit_time = self.current_fream_time
                        self.update_quary.exit_date = self.current_fream_date
                        self.update_quary.save()
                    

        # Draw the FPS on the frame
        self.frame = cv2.putText(self.frame, f'FPS: {self.fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
            if not self.grabbed:
                break

            # Convert the frame to grayscale
            # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
            time.sleep(1 / 30)  # to simulate a frame rate of ~30fps


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# 한상현 연구보고서용 네번째 코드 - 동영상 플레이시 사람이 가까우면 화면 끄기
from scipy.spatial import distance as dist
import numpy as np
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

cap_face = cv2.VideoCapture(0)
cap_movie = cv2.VideoCapture("video.mp4")

eyes = (133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 
        263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398)

while True:

    ret, img_face = cap_face.read()
    ret, img_movie = cap_movie.read()
    if ret is not True:
        break
    img_face_height, img_face_width, _ = img_face.shape
    img_movie_height, img_movie_width, _ = img_movie.shape

    result = face_mesh.process(img_face)

    for facial_landmarks in result.multi_face_landmarks:
        for id in eyes:
            pt1 = facial_landmarks.landmark[id]
            pt_left_eye = facial_landmarks.landmark[133]
            pt_right_eye = facial_landmarks.landmark[362]

            x = int(pt1.x * img_face_width)
            y = int(pt1.y * img_face_height)

            x_left_eye = int(pt_left_eye.x * img_face_width)
            y_left_eye = int(pt_left_eye.y * img_face_height)

            x_right_eye = int(pt_right_eye.x * img_face_width)
            y_right_eye = int(pt_right_eye.y * img_face_height)

            distance = dist.euclidean((x_left_eye, y_left_eye), (x_right_eye, y_right_eye))

            # cv2.putText(img_face, f'Distance: {int(distance)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 2)
            cv2.putText(img_movie, f'Distance: {int(distance)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 2)

            cv2.circle(img_face, (x, y), 2, (0, 0, 255), -1)
            cv2.line(img_face, (x_left_eye, y_left_eye), (x_right_eye, y_right_eye), (0, 0, 255), 1)

            if distance > 55: # 이 값은 임의적이며, 수정 가능함
                img_face = np.zeros((img_face_height, img_face_width, 3), np.uint8)
                img_movie = np.zeros((img_movie_height, img_movie_width, 3), np.uint8)
                cv2.putText(img_movie, f'Distance: {int(distance)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 2)
                cv2.putText(img_movie, f'Stay away, please!', (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 2)

    # cv2.imshow("Image of your face", img_face)
    cv2.imshow("Movie", img_movie)    

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
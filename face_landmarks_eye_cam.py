# 한상현 연구보고서용 세번째 코드 - 웹캠에서 보이는 영상에 양쪽 눈 사이의 거리 표시하기
from scipy.spatial import distance as dist
import numpy as np
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)

eyes = (133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 
        263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398)

while True:

    ret, image = cap.read()
    if ret is not True:
        break
    height, width, _ = image.shape

    result = face_mesh.process(image)

    for facial_landmarks in result.multi_face_landmarks:
        for id in eyes:
            pt1 = facial_landmarks.landmark[id]
            pt_left_eye = facial_landmarks.landmark[133]
            pt_right_eye = facial_landmarks.landmark[362]

            x = int(pt1.x * width)
            y = int(pt1.y * height)

            x_left_eye = int(pt_left_eye.x * width)
            y_left_eye = int(pt_left_eye.y * height)

            x_right_eye = int(pt_right_eye.x * width)
            y_right_eye = int(pt_right_eye.y * height)

            distance = dist.euclidean((x_left_eye, y_left_eye), (x_right_eye, y_right_eye))

            cv2.putText(image, 
                        f'Distance: {int(distance)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            cv2.line(image, (x_left_eye, y_left_eye), (x_right_eye, y_right_eye), (0, 0, 255), 1)

            if distance > 55: # 이 값은 임의적이며, 수정 가능함
                image = np.zeros((height, width, 3), np.uint8)
                # cv2.putText(image, f'Step back, please!', (int(height/2),int(width/2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 2)

    cv2.imshow("Image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


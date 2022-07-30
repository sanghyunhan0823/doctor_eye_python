# 한상현 연구보고서용 첫번째 코드 - 얼굴에 468개 랜드마크 표시하기
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

image_src = cv2.imread("kevin.jpg")
image = cv2.resize(image_src, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
height, width, _ = image.shape

result = face_mesh.process(image)

for facial_landmarks in result.multi_face_landmarks:
    for id in range(0, 468):
        pt1 = facial_landmarks.landmark[id]
        x = int(pt1.x * width)
        y = int(pt1.y * height)

        cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255,0), 1)

cv2.imshow("Image", image)
cv2.waitKey(0)

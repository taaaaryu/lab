import cv2
import numpy as np
import random
import time

def apply_random_blocks(image, rect, block_size=15):
    (x, y, w, h) = rect
    face = image[y:y+h, x:x+w]
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            color = [random.randint(0, 255) for _ in range(3)]
            face[i:i+block_size, j:j+block_size] = color
    image[y:y+h, x:x+w] = face
    return image

def remove_face(image, rect):
    (x, y, w, h) = rect
    image[y:y+h, x:x+w] = (0, 0, 0)
    return image

# カスケードファイルのパス
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# カスケード分類器の読み込み
face_cascade = cv2.CascadeClassifier(cascade_path)

# カメラの起動
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# 顔が検出され続けた時間を記録する辞書
face_timers = {}

# モザイク後に顔を消すまでの時間（秒）
timeout = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケール画像に変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_time = time.time()
    new_face_timers = {}

    # 検出された顔に対する処理
    for (x, y, w, h) in faces:
        face_id = (x, y, w, h)
        if face_id in face_timers:
            # モザイクが一定時間続いた場合、顔を消す
            if current_time - face_timers[face_id] > timeout:
                frame = remove_face(frame, (x, y, w, h))
            else:
                frame = apply_random_blocks(frame, (x, y, w, h))
                new_face_timers[face_id] = face_timers[face_id]
        else:
            # 新たに検出された顔にモザイクをかける
            frame = apply_random_blocks(frame, (x, y, w, h))
            new_face_timers[face_id] = current_time

    face_timers = new_face_timers

    # フレームを表示
    cv2.imshow('Face Random Blocks', frame)

    # 'q'キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

#cap = cv2.VideoCapture('PoseVideos/CheerDance.mp4') #video it is tracking

while True:
    success, img = cap.read()

    if not success:
        print("Can't receive video")
        break

    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    #print(results.pose_landmarks) #shows landmarks, x, y, z and visablitiy)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w ,c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    pTime = 0
    fps = 1/(cTime - pTime)

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

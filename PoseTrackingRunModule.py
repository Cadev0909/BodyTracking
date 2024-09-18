import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('PoseVideos/CheerDance.mp4')  # video it is tracking
pTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()

    if not success:
        print("\nVideo Completed")
        break

    img = detector.findPose(img)
    lmList = detector.getPosistion(img, draw=False)
    if len(lmList) != 0:
        print(lmList[14])  # prints just the landmark point found on mediapipe
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
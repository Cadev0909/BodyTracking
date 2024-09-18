import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, model_comp=1, upBody=False, smooth=True, detCon=.5, minCon=.5):

        self.mode = mode
        self.model_comp = model_comp
        self.upBody = upBody
        self.smooth = smooth
        self.detCon = detCon
        self.minCon = minCon


        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_comp, self.upBody, self.smooth, self.detCon, self.minCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img


    def getPosistion(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w ,c = img.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture('PoseVideos/CheerDance.mp4')  # video it is tracking
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()

        if not success:
            print("\nVideo Completed")
            break

        img = detector.findPose(img)
        lmList = detector.getPosistion(img)
        if len(lmList) != 0:
            print(lmList[14])  # prints just the landmark point found on mediapipe
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0, 255, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
import cv2
import mediapipe as mp
import time
try:
    class handDetector():
        def __init__(self,mode=False,maxhands=2,detectionCon=0.5,trackcon=0.5):
            self.mode = mode
            self.maxhands = maxhands
            self.detectionCon = detectionCon
            self.trackcon = trackcon

            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
            )
            self.mpDraw = mp.solutions.drawing_utils

        def findhands(self,img,draw=True):


            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            #print(results.multi_hand_landmarks)
            if self.results.multi_hand_landmarks:
                for handlms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)

            return img

        def findposition(self,img,handNo=0,draw=True):    
                lmlist = []     
                if self.results.multi_hand_landmarks:
                        myHand = self.results.multi_hand_landmarks[handNo]
                        for id,lm in enumerate(myHand.landmark):
                            print(id,lm)
                            h,w,c = img.shape
                            cx , cy = int(lm.x*w),int(lm.y*h)
                            #print(id,cx,cy)
                            lmlist.append([id,cx,cy])
                            if draw:
                                cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
                return lmlist





    def main():
        pTime = 0
        cTime = 0
        cap = cv2.VideoCapture(0)
        detector = handDetector()
        while True:
            success,img = cap.read()
            detector.findhands(img)
            lmlist = detector.findposition(img)
            if len(lmlist) !=0:
                print(lmlist[4])

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,8,255),3)
            cv2.imshow("image",img)
            cv2.waitKey(1)



    if __name__ == "__main__":
        main()

except Exception as e:
     print(e)
     print("there is an error")
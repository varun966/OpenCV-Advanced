import mediapipe as mp
import cv2 as cv
import time 

cap = cv.VideoCapture(0)

mphand = mp.solutions.hands

hands = mphand.Hands()

mpDraw = mp.solutions.drawing_utils 

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                if id == 4:
                    cv.circle(img, (cx, cy), 15, (255,0,255), cv.FILLED )

                    
            mpDraw.draw_landmarks(img, handlms, mphand.HAND_CONNECTIONS)  # we do not want to draw on RGB 
                                                                          #image as we are displaying BGR image 



    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime 


    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 2)

    cv.imshow('Image', img)
     
    if cv.waitKey(1) & 0xFF == ord('x'):
        break

cv.destroyAllWindows()
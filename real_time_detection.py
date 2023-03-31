import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from keras.models import load_model

mp_drawings = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

    def get_pixel_values(frame):
        frame = cv2.flip(frame, 1)
        
        # Detections
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width, c = img.shape
        
        if results.multi_handedness:
            left_or_right = results.multi_handedness[0].classification[0].label
        
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawings.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
                
            landmarks = []
            for point in mp_hands.HandLandmark:
                normal_landmark = hand.landmark[point]
                landmarks.append(int(normal_landmark.x * width))
                landmarks.append(int(normal_landmark.y * height))
            
            if len(landmarks) != 42:
                pass
        
        return img, landmarks, left_or_right
    

    def detect_sign_language(landmarks, lor):
        """
        Detects which sign language is shown in the image using landmarks and left or right arguments
        """

        model = load_model(r'C:\Users\mp_landmark_model.h5')
        
        if lor == 'Right':
            landmarks.append(1)
        else:
            landmarks.append(0)

        data = np.expand_dims(np.array(landmarks), axis=0)
        pred = np.argmax(model.predict(data))

        return pred




    while 1:

        try:
            ret, frame = cap.read()
            img, landmarks, lor = get_pixel_values(frame)
            print(detect_sign_language(landmarks, lor))

            cv2.imshow("img", img)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        except UnboundLocalError:
            pass

cap.release()
cv2.destroyAllWindows()

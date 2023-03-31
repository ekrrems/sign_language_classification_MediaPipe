from google.protobuf.json_format import MessageToDict
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import os
import random

mp_drawings = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

    def track_fingers(img_path):
        """
        Uses Mediapipe Model to track hands and fingers. Returns an image with landmarks and list of landmark position pixels 
        """
        img = cv2.imread(img_path)

        # Detect hands and landmark positions
        img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width, c = img.shape

        # Find which hand is in the image
        if results.multi_handedness:
            left_or_right = results.multi_handedness[0].classification[0].label
        
        
        if results.multi_hand_landmarks:
            # Draw landmarks on the image
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawings.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
        
            # Find the position of the landmarks
            landmarks = []
            for point in mp_hands.HandLandmark:
                normal_landmark = hand.landmark[point]
                landmarks.append(int(normal_landmark.x * width))
                landmarks.append(int(normal_landmark.y * height))

            if len(landmarks) != 42:
                pass

        return img, landmarks, left_or_right


    def view_random_img(directory_path):
        """
        Chooses a random image from the given directory and applies Mediapipe to track hands and fingers
        """
        # Choose random image from the directory
        subdirs = os.listdir(directory_path)
        rndm_nmbr = random.randint(0, (len(subdirs)-1))
        random_folder = os.path.join(directory_path, subdirs[rndm_nmbr])
        imgs = os.listdir(random_folder)
        random_img_path = os.path.join(random_folder, imgs[random.randint(0, len(imgs)-1)])

        # show the selected image
        img, landmarks, lor = track_fingers(random_img_path)
        print(landmarks)
        print(lor)
    
        cv2.imshow(f"{subdirs[rndm_nmbr]}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    #view_random_img(r"C:\Users\\Train")


    def create_dataset(directory_path):
        """
        Creates a Pandas Dataframe from landmark position values 
        """
        classes = os.listdir(directory_path)
        data = []
        try:
            for c in classes:
                class_path = os.path.join(directory_path, c)
                imgs = os.listdir(class_path)
                for i in imgs:
                    img_path = os.path.join(class_path, i)
                    img, landmark, lor = track_fingers(img_path)
                    landmark.extend([lor, c])
                    data.append(landmark)

        except UnboundLocalError:
            pass
    
        dataset = pd.DataFrame(data, columns=[f'col{n}' for n in range(1,45)])

        return dataset
        

    
    #train_dataset = create_dataset(r"C:\Users\\Train")
    #train_dataset.to_excel("train_data.xlsx")

    #test_dataset = create_dataset(r"C:\Users\\Test")
    #test_dataset.to_excel("test_data.xlsx")



                

                







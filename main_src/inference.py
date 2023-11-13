import cv2
import os
import numpy as np  
import json 

import mediapipe as mp
from tensorflow import keras

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" 
# get_keypoint.py 와 중복된 코드 
"""
mp_holistic = mp.solutions.holistic # pose estimation, face landmarks, hand tracking을 동시에 하는 라이브러리 
# mp_drawing = mp.solutions.drawing_utils # unnecessary

## 0. data 폴더 형식 ## 
DATA_PATH = os.path.join('MPdata')  # 데이터 경로 
actions = np.array(['우리집', '운동장', '월요일'])  # detect할 단어 들 
no_sequences = 30  # 학습에 필요한 하나의 데이터 당 영상 수 
sequence_length = 30  # 영상 frame 수  
start_folder = 30 # 폴더 수 

## 1. inference에 쓸 keypoint 데이터 모으기 ## 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results): 
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

## 4. 영상(이미지 frame) -> 단어 inference 해서 json 형식으로 output ##

# 1. New detection variables
def inferenceToJson(video, ckpt):
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    model = keras.models.load_model(ckpt)
    cap = cv2.VideoCapture(video)
    
    count=0
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            if ret: 
                _, results = mediapipe_detection(frame, holistic)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    # print(actions[np.argmax(res)])
                    predictions.append(actions[np.argmax(res)])
            
            else:
                cap.release() # 영상 파일 사용종료
                cv2.destroyAllWindows()

    
    # sentence to json                 
    return_sentence = np.unique(predictions)
    result = dict()
    if return_sentence:
        result['status'] = "success"
        result['predicted_sentence'] = " ".join(return_sentence)
    else:
        result['status'] = "error"
    
    return result
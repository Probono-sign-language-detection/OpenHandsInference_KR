import cv2
import numpy as np 
import os 
import mediapipe as mp

mp_holistic = mp.solutions.holistic # pose estimation, face landmarks, hand tracking을 동시에 하는 라이브러리 

## 0. data 폴더 형식 ## 
DATA_PATH = os.path.join('MPdata')  # 데이터 경로 
actions = np.array(['우리집', '운동장', '월요일'])  # detect할 단어 들 
no_sequences = 30  # 학습에 필요한 하나의 데이터 당 영상 수 
sequence_length = 30  # 영상 frame 수  
start_folder = 30 # 폴더 수 

## 1. Train에 쓸 keypoint 데이터 모으기 ## 

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


# https://www.futurelearn.com/info/courses/introduction-to-image-analysis-for-plant-phenotyping/0/steps/305359
# 위에 링크 이용해서 영상 -> frame image로 변환 시켜야함 

# mediapipe 모델 세팅 (min_detection_confidence : 탐지 성공 기준값[0.0, 1.0], min_tracking_confidence : 추적 성공 기준값[0.0, 1.0])
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    VIDEO_PATH = os.path.join('FrameData')
    # actions : detect할 단어들 (*np.array)
    for action in actions:
        # no_sequences : 한 단어에 대한 여러개의 학습데이터(영상) 수 (*int)  
        for sequence in range(no_sequences):
            # sequence_length : 하나의 학습데이터(영상)에 대한 frame 수 (*int)
            for frame_num in range(sequence_length):
                cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, action, str(sequence),))
                # Read feed
                ret, frame = cap.read()
                # Make detections
                _, results = mediapipe_detection(frame, holistic)
                 
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()
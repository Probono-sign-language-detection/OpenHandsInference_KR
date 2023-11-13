import numpy as np 
import os 
import mediapipe as mp

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

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


## 2. 학습 Train를 위한 split 및 모델 짜기 ##

# npy 파일(keypoint) 불러오기
label_map = {label:num for num, label in enumerate(actions)} # label_map -> {'우리집': 0, '운동장': 1, '월요일': 2}
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        
X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# LSTM 모델 Build 
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


## 3. 학습 Train 하기 ##
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
# model save
model.save('action.h5')
import cv2
import json
from main_src.inference import inferenceToJson


if __name__ == "__main__" :
    video = 'test_video/운동장.avi'
    ckpt = 'action.h5'
    result = inferenceToJson(video, ckpt)
    print(result)
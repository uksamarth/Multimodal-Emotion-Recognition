"""
File: app_utils.py
Author: U K Samarth
Description: This module contains utility functions for facial expression recognition application.
License: MIT License
"""

import torch
import numpy as np
import mediapipe as mp
from PIL import Image
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image

# Importing necessary components for the Gradio app
from app.model import pth_model_static, pth_model_dynamic, cam, pth_processing
from app.face_utils import get_box, display_info
from app.config import DICT_EMO, config_data
from app.plot import statistics_plot

mp_face_mesh = mp.solutions.face_mesh


def preprocess_image_and_predict(inp):
    inp = np.array(inp)

    if inp is None:
        return None, None

    try:
        h, w = inp.shape[:2]
    except Exception:
        return None, None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(inp)
        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                startX, startY, endX, endY = get_box(fl, w, h)
                cur_face = inp[startY:endY, startX:endX]
                cur_face_n = pth_processing(Image.fromarray(cur_face))
                with torch.no_grad():
                    prediction = (
                        torch.nn.functional.softmax(pth_model_static(cur_face_n), dim=1)
                        .detach()
                        .numpy()[0]
                    )
                confidences = {DICT_EMO[i]: float(prediction[i]) for i in range(7)}
                grayscale_cam = cam(input_tensor=cur_face_n)
                grayscale_cam = grayscale_cam[0, :]
                cur_face_hm = cv2.resize(cur_face,(224,224))
                cur_face_hm = np.float32(cur_face_hm) / 255
                heatmap = show_cam_on_image(cur_face_hm, grayscale_cam, use_rgb=True)

    return cur_face, heatmap, confidences


def preprocess_video_and_predict(video):

    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = np.round(cap.get(cv2.CAP_PROP_FPS))

    path_save_video_face = 'result_face.mp4'
    vid_writer_face = cv2.VideoWriter(path_save_video_face, cv2.VideoWriter_fourcc(*'mp4v'), fps, (224, 224))

    path_save_video_hm = 'result_hm.mp4'
    vid_writer_hm = cv2.VideoWriter(path_save_video_hm, cv2.VideoWriter_fourcc(*'mp4v'), fps, (224, 224))

    lstm_features = []
    count_frame = 1
    count_face = 0
    probs = []
    frames = []
    last_output = None
    last_heatmap = None 
    cur_face = None

    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            _, frame = cap.read()
            if frame is None: break

            frame_copy = frame.copy()
            frame_copy.flags.writeable = False
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_copy)
            frame_copy.flags.writeable = True

            if results.multi_face_landmarks:
                for fl in results.multi_face_landmarks:
                    startX, startY, endX, endY  = get_box(fl, w, h)
                    cur_face = frame_copy[startY:endY, startX: endX]

                    if count_face%config_data.FRAME_DOWNSAMPLING == 0:
                        cur_face_copy = pth_processing(Image.fromarray(cur_face))
                        with torch.no_grad():
                            features = torch.nn.functional.relu(pth_model_static.extract_features(cur_face_copy)).detach().numpy()

                        grayscale_cam = cam(input_tensor=cur_face_copy)
                        grayscale_cam = grayscale_cam[0, :]
                        cur_face_hm = cv2.resize(cur_face,(224,224), interpolation = cv2.INTER_AREA)
                        cur_face_hm = np.float32(cur_face_hm) / 255
                        heatmap = show_cam_on_image(cur_face_hm, grayscale_cam, use_rgb=False)
                        last_heatmap = heatmap
        
                        if len(lstm_features) == 0:
                            lstm_features = [features]*10
                        else:
                            lstm_features = lstm_features[1:] + [features]

                        lstm_f = torch.from_numpy(np.vstack(lstm_features))
                        lstm_f = torch.unsqueeze(lstm_f, 0)
                        with torch.no_grad():
                            output = pth_model_dynamic(lstm_f).detach().numpy()
                        last_output = output

                        if count_face == 0:
                            count_face += 1

                    else:
                        if last_output is not None:
                            output = last_output
                            heatmap = last_heatmap

                        elif last_output is None:
                            output = np.empty((1, 7))
                            output[:] = np.nan
                            
                    probs.append(output[0])
                    frames.append(count_frame)
            else:
                if last_output is not None:
                    lstm_features = []
                    empty = np.empty((7))
                    empty[:] = np.nan
                    probs.append(empty)
                    frames.append(count_frame)                        

            if cur_face is not None:
                heatmap_f = display_info(heatmap, 'Frame: {}'.format(count_frame), box_scale=.3)

                cur_face = cv2.cvtColor(cur_face, cv2.COLOR_RGB2BGR)
                cur_face = cv2.resize(cur_face, (224,224), interpolation = cv2.INTER_AREA)
                cur_face = display_info(cur_face, 'Frame: {}'.format(count_frame), box_scale=.3)
                vid_writer_face.write(cur_face)
                vid_writer_hm.write(heatmap_f)

            count_frame += 1
            if count_face != 0:
                count_face += 1

        vid_writer_face.release()
        vid_writer_hm.release()

        stat = statistics_plot(frames, probs)

        if not stat:
            return None, None, None, None
        
    return video, path_save_video_face, path_save_video_hm, stat
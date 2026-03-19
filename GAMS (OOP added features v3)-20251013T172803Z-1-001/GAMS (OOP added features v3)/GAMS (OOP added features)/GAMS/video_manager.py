import pygame
import time
import sys
import os
import cv2
import numpy as np

from utils import resources_path, get_pygame_window_pos

class VideoManager:
    def __init__(self, win_size, win, pose_detector, scale, audio_manager=None):
        self.win_size = win_size
        self.win = win
        self.pose_detector = pose_detector
        self.scale = scale
        self.audio_manager = audio_manager

        # Video paths
        self.introvid_path = os.path.join(resources_path, "bookends", "introduction.mp4")
        self.chocks_inserted_video = os.path.join(resources_path, "bookends", "chocks_inserted.mp4")

    def play_introduction_video(self):
        # Play bookends audio first (assuming audio is handled elsewhere)
        # self.play_bookends_audio("introduction")  # Moved to audio manager

        intro_path = getattr(self, "introvid_path", None)
        if not intro_path or not os.path.exists(intro_path):
            intro_path = os.path.join(resources_path, "bookends", "introduction.mp4")
            if not os.path.exists(intro_path):
                print("[Bookends] introduction.mp4 not found.")
                return

        # Use cv2 to play the video file
        cap = cv2.VideoCapture(intro_path)
        if not cap.isOpened():
            print(f"[Bookends] Failed to open introduction.mp4: {intro_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps) if fps and fps > 0 else 30
        delay = int(1000 / fps)

        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = vid_width / vid_height if vid_height != 0 else 1.0

        win_width, win_height = self.win.get_size()
        scaled_height = win_height
        scaled_width = int(scaled_height * aspect_ratio)

        if scaled_width > win_width:
            scaled_width = win_width
            scaled_height = int(scaled_width / aspect_ratio)

        cv2.namedWindow("Introduction", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Introduction", scaled_width, scaled_height)

        pygamewin_pos = get_pygame_window_pos()
        x = pygamewin_pos[0] + (win_width - scaled_width) // 2
        y = pygamewin_pos[1]  # Position at the top of the pygame window
        cv2.moveWindow("Introduction", x, y)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Introduction", frame)
            key = cv2.waitKey(delay)
            if key == 32:  # Space key
                if self.audio_manager:
                    self.audio_manager.stop()
                break

        cap.release()
        cv2.destroyWindow("Introduction")

    def play_chocksinserted_video(self, pygamewin_pos):
        if not getattr(self, "chocks_inserted_video", None):
            print("[Chocks Inserted] No video loaded.")
            return

        cap = cv2.VideoCapture(self.chocks_inserted_video)
        if not cap.isOpened():
            print(f"[Chocks Inserted] Failed to open video: {self.chocks_inserted_video}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps) if fps and fps > 0 else 30
        delay = int(1000 / fps)

        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = vid_width / vid_height if vid_height != 0 else 1.0

        win_width, win_height = self.win.get_size()
        scaled_height = win_height
        scaled_width = int(scaled_height * aspect_ratio)

        if scaled_width > win_width:
            scaled_width = win_width
            scaled_height = int(scaled_width / aspect_ratio)

        cv2.namedWindow("Chocks Inserted", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chocks Inserted", scaled_width, scaled_height)

        x = max(0, pygamewin_pos[0] - scaled_width - 5)
        y = pygamewin_pos[1]
        cv2.moveWindow("Chocks Inserted", x, y)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Chocks Inserted", frame)
            if cv2.waitKey(delay) == -1:
                pass

        cap.release()
        cv2.destroyWindow("Chocks Inserted")

import cv2
import os
from pathlib import Path

class VideoReader:
    def __init__(self, video_path):
        self.path = str(video_path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Video not found: {self.path}")

        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.path}")

        #读取元数据
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int (self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        """支持 for frame in video: 写法"""
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def get_info(self):
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames
        }

    def close(self):
        self.cap.release()

class VideoWriter:
    def __init__(self, output_path, width, height, fps):
        self.path = str(output_path)
        self.width = width
        self.height = height
        self.fps = fps

        #确保输出目录的存在
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        #初始化写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.path, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)

    def close(self):
        self.writer.release()
        print(f"[VideoWriter] Saved to {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def save_comparison_video(output_path, raw_frames, processed_frames, fps=30):
    """
    工具函数：快速保存左右对比视频
    """
    if len(raw_frames) == 0:
        return

    h, w, _ = raw_frames[0].shape
    #宽度翻倍（左右拼接）
    with VideoWriter(output_path, w*2, h, fps) as out:
        min_len = min(len(raw_frames), len(processed_frames))
        for i in range(min_len):
            #横向拼接
            combined = cv2.hconcat(raw_frames[i], processed_frames[i])
            out.write(combined)
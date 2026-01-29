import sys
from fileinput import close
import cv2
from pathlib import Path
from tqdm import tqdm


current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
sys.path.append(str(project_root / "src"))

from pose_solver.core.config import Config
from pose_solver.data.video_io import VideoReader

def extract_frames(video_path_str):
    video_path = Path(video_path_str)

    #输出目录
    base_dir = Config.VIDEO_FRAMES_DIR
    save_dir = base_dir / f"{video_path.stem}_frame"

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"创建目录：{save_dir}")
    else:
        print(f"目录已存在：{save_dir}(文件可能会被覆盖)")

    # 读取视频
    try:
        reader = VideoReader(video_path)
    except Exception as e:
        print(f"[错误]{e}")
        return
    print(f"正在提取：{video_path.name}")
    print(f"目标路径：{save_dir}")

    # 逐帧保存
    pbar = tqdm(total=reader.total_frames, unit="img")
    count = 0
    for frame in reader:
        file_name = f"{count:06d}.jpg"
        cv2.imwrite(str(save_dir/file_name), frame)
        count += 1
        pbar.update(1)

    pbar.close()
    reader.close()
    print(f"完成！一共{count}张照片。")

if __name__ == "__main__":
    target_video = Config.VIDEO_PATH
    extract_frames(target_video)
import os
import cv2
from pytube import YouTube

def download_video(youtube_url, output_path):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension='mp4').first()
    stream.download(output_path=output_path, filename='video.mp4')
    return os.path.join(output_path, 'video.mp4')

def process_video(video_path, output_folder, interval=10, target_size=(640, 640)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    current_time = 0
    frame_number = 0
    
    while current_time < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        success, frame = cap.read()
        if not success:
            break
        
        height, width = frame.shape[:2]
        middle = width // 2
        
        left_half = frame[:, :middle]
        right_half = frame[:, middle:]
        
        left_resized = cv2.resize(left_half, target_size)
        right_resized = cv2.resize(right_half, target_size)
        
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_number}_left.jpg"), left_resized)
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_number}_right.jpg"), right_resized)
        
        current_time += interval
        frame_number += 1
    
    cap.release()

def main():
    youtube_url = 'https://www.youtube.com/watch?v=A0cjGAhyLtk&ab_channel=Vigevano1955'
    output_folder = './output_frames'
    
    video_path = download_video(youtube_url, output_folder)
    process_video(video_path, output_folder)

if __name__ == "__main__":
    main()

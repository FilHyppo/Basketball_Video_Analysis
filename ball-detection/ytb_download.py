import os
import cv2
import subprocess


def scarica_video(output_dir, youtube_url):
    output_path = os.path.join(output_dir, 'video.mp4')
    comando = ['yt-dlp', '-o', output_path, youtube_url]
    subprocess.run(comando)

    return output_path

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
    youtube_url = 'https://www.youtube.com/watch?v=-zcIsk6GE6Q'
    video_output_dir = './videos_to_annotate'    
    video_path = scarica_video(video_output_dir, youtube_url)
    

if __name__ == "__main__":
    main()

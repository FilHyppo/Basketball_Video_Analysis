import os
import cv2
import subprocess


def process_video(video_path, output_folder, interval=10, target_size=(640, 640)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    tot_number_of_frames = int(duration / interval) * 2
    #Calculate the number of digits needed to represent the total number of frames
    tot_number_of_frames = len(str(tot_number_of_frames))

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
        
        video_name = os.path.basename(video_path)

        frame_id = f"{frame_number:0{tot_number_of_frames}d}"
        cv2.imwrite(os.path.join(output_folder, f"{video_name}_{frame_id}_left.jpg"), left_resized)
        cv2.imwrite(os.path.join(output_folder, f"{video_name}_{frame_id}_right.jpg"), right_resized)
        
        current_time += interval
        frame_number += 1
    
    cap.release()


def main():
    #youtube_url = 'https://www.youtube.com/watch?v=-zcIsk6GE6Q'
    #video_output_dir = './videos_to_annotate'    
    #video_path = scarica_video(video_output_dir, youtube_url)
    frames_output_dir = './images_to_annotate'
    videos_folder = 'C:\\Users\\marco\\Downloads\\wetransfer_novellara-magik_2023-11-20_1424'
    for video_file in os.listdir(videos_folder):
        video_path = os.path.join(videos_folder, video_file)
        process_video(video_path, frames_output_dir, interval=5)

if __name__ == "__main__":
    main()

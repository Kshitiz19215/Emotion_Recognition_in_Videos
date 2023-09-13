import cv2
import os

video_data_path = 'data/emotion_videos/'

def extract_frames(video_file, output_folder):
    try:
        cap = cv2.VideoCapture(video_file)
        os.makedirs(output_folder, exist_ok=True)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)

            frame_count += 1
    
        cap.release()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def preprocess_frame(frame):
    try:
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = frame / 255.0
        
        return frame
    except Exception as e:
        print(f"An error occurred during frame preprocessing: {str(e)}")

video_file = 'C:/Users/ashar/Desktop/Emotion_Recognition/data/vid.mp4'
output_folder = 'data/frames/'
extract_frames(video_file, output_folder)
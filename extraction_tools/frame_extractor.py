import cv2
import numpy as np
import os

def extract_frames(video_path, output_folder, num_parts=12, frames_per_part=20):
    """
    Extract frames from a video file, dividing the video into equal parts and extracting frames at equal intervals from each part.
    
    :param video_path: Path to the video file
    :param output_folder: Folder where extracted frames will be saved
    :param num_parts: Number of parts to divide the video into
    :param frames_per_part: Number of frames to extract from each part
    :return: None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get total number of frames and video duration
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"Total frames: {total_frames}, FPS: {fps}, Duration: {duration}s")

    # Calculate intervals
    part_duration = duration / num_parts
    frames_interval = total_frames / (num_parts * frames_per_part)

    print(f"Part duration: {part_duration}s, Frames interval: {frames_interval}")

    # Extract frames
    for part in range(num_parts):
        for frame_num in range(frames_per_part):
            frame_id = int((part * part_duration * fps) + (frame_num * frames_interval * fps))

            # Set video to the correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()

            if not ret:
                print(f"Error reading frame {frame_id}")
                continue

            # Save the frame
            frame_filename = f"{output_folder}/frame_part{part}_num{frame_num}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")

    # Release the video capture object
    cap.release()
    print("Extraction complete.")

def main():
    # Script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Main project directory
    project_dir = os.path.join(script_dir, '..')

    # Get movie name
    movie_name = input("Choose movie to be extracted (without extension): ")

    # Get extension
    extension = input("Choose extension (mp4, avi, etc): ")

    # Path to the video file
    video_path = os.path.join(project_dir, f"data/movies/{movie_name}.{extension}")

    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: The file {video_path} does not exist.")
        return

    # Check if the file is a valid video file
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: The file {video_path} is not a valid video file or the format is not supported.")
            return
        cap.release()
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Folder where extracted frames will be saved
    output_folder = os.path.join(project_dir, f"data/frames/{movie_name}")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Number of parts to divide the video into
    num_parts = 12

    # Number of frames to extract from each part
    frames_per_part = 20

    # Extract frames
    extract_frames(video_path, output_folder, num_parts, frames_per_part)

# Don't forget to define the extract_frames function here or import it if it's defined in another file

# Run the main function
if __name__ == "__main__":
    main()
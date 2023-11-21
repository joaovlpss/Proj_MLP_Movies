import cv2
import os
import numpy as np

def preprocess_frame(frame):
    """
    Apply preprocessing steps to a frame
    """
    # Resize the frame
    frame = cv2.resize(frame, (256, 256))

    # Color normalization (disabled for now)
    frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Noise reduction (Gaussian Blur)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return frame

def preprocess_movie_frames(movie_name, input_folder, output_folder):
    """
    Preprocess all frames of a movie
    """
    input_path = os.path.join(input_folder, movie_name)
    output_path = os.path.join(output_folder, movie_name)

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process each frame
    for frame_file in os.listdir(input_path):
        frame_path = os.path.join(input_path, frame_file)
        frame = cv2.imread(frame_path)
        processed_frame = preprocess_frame(frame)
        
        # Save the preprocessed frame
        cv2.imwrite(os.path.join(output_path, frame_file), processed_frame)

def main():
    # Main project directory
    project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    for x in os.scandir(os.path.join(project_dir, 'data/frames')):
        if x.is_dir():
            print(f"Processing {x.name}...")
            preprocess_movie_frames(x.name, os.path.join(project_dir, 'data/frames'), os.path.join(project_dir, 'data/preprocessed_frames'))

if __name__ == "__main__":
    main()

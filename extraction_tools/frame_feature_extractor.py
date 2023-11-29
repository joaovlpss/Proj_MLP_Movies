import os
import cv2
import numpy as np
import mahotas
from matplotlib import pyplot as plt

def process_frame(frame):
    """Process a single frame to obtain HSV histogram and Haralick features."""
    # Convert to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate histograms for each channel with 32 bins
    hist_h = cv2.calcHist([hsv_frame], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv_frame], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv_frame], [2], None, [32], [0, 256])

    # Normalize histograms
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    # Calculate Haralick features
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haralick_features = mahotas.features.haralick(gray_frame).mean(axis=0)

    return hist_h, hist_s, hist_v, haralick_features

def process_movie(movie_name):
    """Process all frames of a movie."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))

    frame_dir = os.path.join(project_dir, 'data', 'preprocessed_frames', movie_name)
    histogram_dir = os.path.join(project_dir, 'data', 'histograms', movie_name)
    haralick_dir = os.path.join(project_dir, 'data', 'haralick_features', movie_name)

    # Create directories if they don't exist
    os.makedirs(histogram_dir, exist_ok=True)
    os.makedirs(haralick_dir, exist_ok=True)

    for frame_file in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)

        hist_h, hist_s, hist_v, haralick_features = process_frame(frame)

        # Save histograms and Haralick features
        hist_file = os.path.join(histogram_dir, f'hist_{frame_file}.npy')
        haralick_file = os.path.join(haralick_dir, f'haralick_{frame_file}.npy')

        np.save(hist_file, np.hstack([hist_h, hist_s, hist_v]))
        np.save(haralick_file, haralick_features)

def main():
    # Main project directory
    project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    for x in os.scandir(os.path.join(project_dir, 'data/preprocessed_frames')):
        if x.is_dir():
            print(f"Processing {x.name}...")
            process_movie(x.name)


if __name__ == "__main__":
    main()
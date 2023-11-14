# MLP Movie Genre Classifier Project

This project aims to develop a machine learning classifier capable of categorizing movies into genres based on their visual content. It leverages the power of Multi-Layer Perceptrons (MLPs) to analyze and classify movies.

## Project Structure

The project is organized into several directories, each serving a specific purpose:

- `data/`: Stores movies and frames
  - `frames/`: Contains extracted frames for each movie, organized in subfolders by movie names (e.g., `scream`)
  - `movies/`: Holds all movie files in formats like .mkv, .mp4, etc.
- `extraction_tools/`: Hosts tools for frame extraction and preprocessing
- `models/`: Will host the model trainer and related scripts
- `outputs/`: For storing all trained models
- `src/`
  - `visualization_tools/`: Will contain tools for visualizing frame information, data, and MLP training progress

## Key Components

### 1. Frame Extractor (`frame_extractor.py`)

This script divides a movie into equal parts and extracts a specified number of frames from each part. It helps in gathering a diverse set of frames from across the movie for analysis.

#### Usage

1. Run the script and input the name of the movie (without extension).
2. Specify the file extension (e.g., mkv, mp4).
3. The script processes the movie and saves the frames in the `data/frames/{movie_name}` directory.

### 2. Frame Preprocessor (`frame_preprocessor.py`)

This script preprocesses the extracted frames by resizing, normalizing, and reducing noise. It standardizes the frames for more consistent analysis and input to the MLP.

#### Usage

1. Run the script and input the name of the movie for which you want to preprocess frames.
2. The script reads frames from `data/frames/{movie_name}`, processes them, and saves the output in `data/preprocessed_frames/{movie_name}`.

## Getting Started

1. Place your movie files in the `data/movies/` directory.
2. Run `frame_extractor.py` to extract frames from the movies.
3. Run `frame_preprocessor.py` to preprocess the extracted frames.

## Upcoming Features

- Model training scripts in the `models/` directory.
- Visualization tools for analyzing frame features and model performance.
- Audio freature extractions and preprocessing in the `extraction_tools/` directory.

This project is under development and more features will be added.

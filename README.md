# MLP Movie Genre Classifier Project

## Overview
This project aims to develop a machine learning classifier capable of categorizing movies into genres based on their visual content. It leverages the power of Multi-Layer Perceptrons (MLPs) to analyze and classify movies by extracting and analyzing frame features.

## Project Structure

### Directories
- `data/`: Contains movies and frames.
  - `frames/`: Extracted frames for each movie, organized by movie names.
  - `movies/`: Movie files in formats like .mkv, .mp4, etc.
- `extraction_tools/`: Scripts for frame extraction and preprocessing.
- `models/`: Contains the MLP model training scripts and related utilities.
- `outputs/`: Storage for trained models and outputs.
- `src/`: Source code for the project.

## Key Components

### Extraction Tools
1. **Frame Extractor (`frame_extractor.py`)**: Divides movies into equal parts, extracts frames for diverse analysis.
2. **Frame Preprocessor (`frame_preprocessor.py`)**: Standardizes frames by resizing, normalizing, and reducing noise.

### Model Preparation
- **Feature Input Creator (`feature_input_creator.py`)**: Aggregates features into a consistent format for the MLP model.

### Visualization Tools
- **Feature Visualization (`feature_visualization.py`)**: GUI for selecting and displaying histograms and feature data.

## Usage Instructions

1. Place movies in `data/movies/`.
2. Run `frame_extractor.py` to extract frames.
3. Use `frame_preprocessor.py` to preprocess frames.
4. Aggregate features using `feature_input_creator.py`.
5. Visualize features with `feature_visualization.py`.

## Upcoming Features
- Enhanced model training scripts.
- Advanced visualization tools for frame features and model performance.
- Audio feature extraction and preprocessing.
import numpy as np
import os

def load_and_average_features(feature_dir, feature_type=""):
    """Load all features from the given directory and average them."""
    feature_files = [file for file in sorted(os.listdir(feature_dir)) if file.endswith('.npy')]
    features = []
    for file in feature_files:
        feature_path = os.path.join(feature_dir, file)
        feature_data = np.load(feature_path)
        features.append(feature_data)

    if features:
        features_array = np.array(features, dtype=object)  # Use object dtype to handle different lengths
        averaged_features = np.array([np.mean(feature_array) for feature_array in features_array.T])
        print(f"Averaged {feature_type} features length: {len(averaged_features)}")
        return averaged_features
    else:
        print(f"No {feature_type} features found in {feature_dir}")
        return np.array([])

def create_feature_input(movie_name, project_dir, genre_label):
    # Paths for feature directories
    histogram_dir = os.path.join(project_dir, 'data', 'histograms', movie_name)
    haralick_dir = os.path.join(project_dir, 'data', 'haralick_features', movie_name)
    audio_feature_dir = os.path.join(project_dir, 'data', 'audio_features', movie_name)
    frequency_feature_dir = os.path.join(project_dir, 'data', 'frequency_features', movie_name)
    output_dir = os.path.join(project_dir, 'data', 'combined_features')

    # Load and average features
    histograms = load_and_average_features(histogram_dir, "HSV Histogram")
    haralick_features = load_and_average_features(haralick_dir, "Haralick")
    audio_features = load_and_average_features(audio_feature_dir, "Audio")
    frequency_features = load_and_average_features(frequency_feature_dir, "Frequency")

    # Combine features
    combined_features = np.concatenate((histograms, haralick_features, audio_features, frequency_features))

    # Normalize features
    normalized_features = (combined_features - np.min(combined_features)) / (np.max(combined_features) - np.min(combined_features))

    # One-hot encode genre label
    label_map = {'horror': [1, 0, 0, 0], 'sci-fi': [0, 1, 0, 0], 'comedy': [0, 0, 1, 0], 'fantasy': [0, 0, 0, 1]}
    one_hot_label = label_map.get(genre_label, [0, 0, 0, 0])

    # Add label to the feature vector
    final_feature_vector = np.hstack((one_hot_label, normalized_features))

    # Save combined features with label
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{movie_name}_features_with_label.npy')
    np.save(output_path, final_feature_vector)

    print(f"Saved feature vector for {movie_name} with size {final_feature_vector.shape[0]}")

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Create feature input for each movie
    for file in os.listdir(os.path.join(project_dir, 'data', 'movies')):
        movie_name = '.'.join(file.split('.')[:-1])
        genre_label = movie_name.split('_')[0]

        print(f"Creating feature input for {movie_name}...")
        create_feature_input(movie_name, project_dir, genre_label)

    print("Feature input creation complete.")

if __name__ == '__main__':
    main()

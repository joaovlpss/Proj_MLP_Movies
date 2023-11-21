import numpy as np
import os

def load_and_aggregate_features(movie_name, feature_dir):
    """Load all features from the given directory and aggregate them."""
    feature_files = sorted(os.listdir(feature_dir))
    features = [np.load(os.path.join(feature_dir, file)) for file in feature_files]
    aggregated_features = np.mean(np.array(features), axis=0)
    return aggregated_features

def create_feature_input(movie_name, project_dir, genre_label):
    # Paths for feature directories
    histogram_dir = os.path.join(project_dir, 'data', 'histograms', movie_name)
    haralick_dir = os.path.join(project_dir, 'data', 'haralick_features', movie_name)
    output_dir = os.path.join(project_dir, 'data', 'combined_features')

    # Load and aggregate features
    histograms = load_and_aggregate_features(movie_name, histogram_dir)
    haralick_features = load_and_aggregate_features(movie_name, haralick_dir)

    # Combine features
    combined_features = np.hstack((histograms, haralick_features))

    # Normalize features
    combined_features = (combined_features - np.min(combined_features)) / (np.max(combined_features) - np.min(combined_features))

    # One-hot encode genre label (order: horror, sci-fi, comedy, fantasy)
    label_map = {'horror': [1, 0, 0, 0], 'sci-fi': [0, 1, 0, 0], 'comedy': [0, 0, 1, 0], 'fantasy': [0, 0, 0, 1]}
    one_hot_label = label_map.get(genre_label, [0, 0, 0, 0])

    # Add label to the feature vector
    final_feature_vector = np.hstack((one_hot_label, combined_features))

    # Save combined features with label
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{movie_name}_features_with_label.npy')
    np.save(output_path, final_feature_vector)

    print(f"Feature vector with label saved to {output_path}")
    print(f"Size of feature vector: {final_feature_vector.shape[0]}")



def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Create feature input for each movie
    for file in os.listdir(os.path.join(project_dir, 'data', 'movies')):
        movie_name = '.'.join(file.split('.')[:-1])
        genre_label = movie_name.split('_')[0]
        movie_name = movie_name.split('_')[1]

        print(f"Creating feature input for {movie_name}...")
        create_feature_input(movie_name, project_dir, genre_label)

    print("Feature input creation complete.")


if __name__ == '__main__':
    main()

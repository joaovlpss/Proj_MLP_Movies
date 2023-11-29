import librosa
import numpy as np
import os

class AudioAnalyzer:
    """
    A class for analyzing audio files, focusing on features like amplitude envelope and zero-crossing rate,
    and fitting these features to statistical distributions.
    """

    def __init__(self, file_paths):
        """
        Initialize the AudioAnalyzer with a list of audio file paths.
        :param file_paths: List of file paths to audio files.
        """
        self.file_paths = file_paths

    @staticmethod
    def amplitude_envelope(signal, frame_length, hop_length):
        """
        Calculate the amplitude envelope of an audio signal.
        :param signal: Audio signal array.
        :param frame_length: Length of each frame.
        :param hop_length: Hop length between frames.
        :return: Numpy array of maximum amplitude for each frame.
        """
        return np.array([max(signal[i:i+frame_length]) for i in range(0, len(signal), hop_length)])

    def load_audio(self, position):
        """
        Load an audio file from the specified position in the file_paths list.
        :param position: Index of the file in the file_paths list.
        :return: Tuple of (audio signal, sample rate).
        """
        try:
            return librosa.load(self.file_paths[position])
        except IndexError:
            print(f"No file found at position {position}.")
            return None, None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None, None

    def extract_and_save_features(self, position, feature_dir, feature_func, frame_length=1024, hop_length=512):
        """
        Extract audio features and save them as a .npy file.
        :param position: Index of the file in the file_paths list.
        :param feature_dir: Directory to save the feature files.
        :param feature_func: Function to compute the audio feature.
        :param frame_length: Length of each frame.
        :param hop_length: Hop length between frames.
        """
        signal, sr = self.load_audio(position)
        if signal is None:
            return

        feature = feature_func(signal, frame_length, hop_length)
        feature_file = os.path.join(feature_dir, f'audio_feature_{os.path.basename(self.file_paths[position])}.npy')
        np.save(feature_file, feature)
        print(f"Saved audio feature for {os.path.basename(self.file_paths[position])}")

def main():
    # Define the folder paths
    project_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    audio_folder = os.path.join(project_folder, "data/audios")
    feature_folder = os.path.join(project_folder, "data/audio_features")

    # Ensure feature folder exists
    os.makedirs(feature_folder, exist_ok=True)

    # List all .wav files in the audio folder
    file_paths = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith('.wav')]

    # Initialize the AudioAnalyzer
    analyzer = AudioAnalyzer(file_paths)

    # Process each audio file
    for i in range(len(file_paths)):
        # Extract and save audio features as .npy files
        analyzer.extract_and_save_features(i, feature_folder, AudioAnalyzer.amplitude_envelope)

if __name__ == "__main__":
    main()
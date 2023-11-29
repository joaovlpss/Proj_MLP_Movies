import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

class FrequencyAnalyzer:
    def __init__(self, audio_folder, output_folder):
        self.audio_folder = audio_folder
        self.output_folder = output_folder

    @staticmethod
    def calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate):
        frequency_range = sample_rate / 2
        frequency_delta = frequency_range / spectrogram.shape[0]
        return int(np.floor(split_frequency / frequency_delta))

    @staticmethod
    def calculate_band_energy_ratio(spectrogram, split_frequency, sample_rate):
        split_bin = FrequencyAnalyzer.calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate)
        power_spec = np.abs(spectrogram) ** 2
        band_energy_ratio = [frame[:split_bin].sum() / frame[split_bin:].sum() for frame in power_spec.T if frame[split_bin:].sum() != 0]
        return np.array(band_energy_ratio)

    def process_audio_file(self, file_path):
        signal, sr = librosa.load(file_path)
        spectrogram = librosa.stft(signal, n_fft=2048, hop_length=512)
        band_energy_ratio = self.calculate_band_energy_ratio(spectrogram, 2000, sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]

        return band_energy_ratio, spectral_centroid

    def plot_and_save_features(self, file_path):
        try:
            file_name = os.path.basename(file_path)
            output_file_path = os.path.join(self.output_folder, f"features_{file_name}.png")

            band_energy_ratio, spectral_centroid = self.process_audio_file(file_path)
            
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.plot(band_energy_ratio)
            plt.title('Band Energy Ratio')
            plt.xlabel('Frames')
            plt.ylabel('Ratio')

            plt.subplot(1, 2, 2)
            plt.plot(spectral_centroid)
            plt.title('Spectral Centroid')
            plt.xlabel('Frames')
            plt.ylabel('Centroid Frequency')

            plt.savefig(output_file_path)
            plt.close()
            print(f"Processed and saved features for {file_name}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    project_folder = os.path.join(os.path.abspath(__file__), "..")
    audio_folder = os.path.join(project_folder, "data/audios")
    output_folder = os.path.join(project_folder, "data/audio_features")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    analyzer = FrequencyAnalyzer(audio_folder, output_folder)
    for file in os.listdir(audio_folder):
        if file.endswith('.wav'):
            file_path = os.path.join(audio_folder, file)
            analyzer.plot_and_save_features(file_path)

if __name__ == "__main__":
    main()

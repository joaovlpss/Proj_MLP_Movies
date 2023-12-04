import os
import librosa
import numpy as np

def calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate):
    frequency_range = sample_rate / 2
    frequency_delta = frequency_range / spectrogram.shape[0]
    split_frequency_bin = np.floor(split_frequency / frequency_delta)
    return int(split_frequency_bin)

def calculate_band_energy_ratio(spectrogram, split_frequency, sample_rate):
    split_frequency_bin = calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate)
    power_spec = np.abs(spectrogram)**2
    power_spec = power_spec.T
    band_energy_ratio = []
    for frame in power_spec:
        sum_low = frame[:split_frequency_bin].sum()
        sum_high = frame[split_frequency_bin:].sum()
        if sum_high != 0:
            band_energy_ratio.append(sum_low / sum_high)
    return np.array(band_energy_ratio)

def calculate_histogram(feature, num_bins=32):
    hist, _ = np.histogram(feature, bins=num_bins, density=True)
    return hist.flatten()

def process_audio_file(file_path, num_bins=32):
    signal, sr = librosa.load(file_path)
    spectrogram = librosa.stft(signal, n_fft=2048, hop_length=512)

    # Calculate Band Energy Ratio
    band_energy_ratio = calculate_band_energy_ratio(spectrogram, 2000, sr)
    energy_ratio = np.sum(band_energy_ratio > 10000) / np.sum(band_energy_ratio <= 10000)

    # Calculate Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]

    # Calculate histogram for Spectral Centroid
    spectral_centroid_hist = calculate_histogram(spectral_centroid, num_bins)

    return energy_ratio, spectral_centroid_hist

def main():
    project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    audio_folder = os.path.join(project_folder, "data/audios")
    output_folder = os.path.join(project_folder, "data/frequency_features")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(audio_folder):
        if file_name.endswith('.wav'):
            file_name_without_extension = os.path.splitext(file_name)[0]
            file_results_folder = os.path.join(output_folder, file_name_without_extension)
            if os.path.exists(file_results_folder):
                continue
            if not os.path.exists(file_results_folder):
                os.makedirs(file_results_folder)

            file_path = os.path.join(audio_folder, file_name)
            energy_ratio, spectral_centroid_hist = process_audio_file(file_path)

            # Save results as .npy file
            results = np.array([energy_ratio, *spectral_centroid_hist])
            np.save(os.path.join(file_results_folder, "frequency_analysis_results.npy"), results)

if __name__ == "__main__":
    main()

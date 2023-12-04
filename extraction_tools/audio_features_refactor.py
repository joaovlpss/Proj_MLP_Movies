import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto, skewnorm

def amplitude_envelope(signal, frame_length, hop_length):
    return np.array([max(signal[i:i+frame_length]) for i in range(0, signal.size, hop_length)])

def zero_crossing_rate(signal, frame_length, hop_length):
    return librosa.feature.zero_crossing_rate(signal, frame_length=frame_length, hop_length=hop_length, center=True)[0]

def load_audio(file_path):
    try:
        return librosa.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def calculate_histogram(feature, num_bins=32):
    hist, _ = np.histogram(feature, bins=num_bins, density=True)
    return hist.flatten()

def process_audio_files(audio_folder, num_bins=32):
    results = []
    for file in os.listdir(audio_folder):
        if file.endswith('.wav'):
            file_name = os.path.splitext(file)[0]
            file_path = os.path.join(audio_folder, file)
            signal, sr = load_audio(file_path)

            if signal is not None:
                frame_length = 1024
                hop_length = 512

                # Calculate histograms
                amp_env = amplitude_envelope(signal, frame_length, hop_length)
                zcr = zero_crossing_rate(signal, frame_length, hop_length)

                amp_env_hist = calculate_histogram(amp_env, num_bins)
                zcr_hist = calculate_histogram(zcr, num_bins)

                combined_features = np.hstack([amp_env_hist, zcr_hist])
                results.append((file_name, combined_features))

    return results

def write_results(results, output_folder):
    for file_name, file_results in results:
        if (not os.path.exists(os.path.join(output_folder, f"{file_name}"))):
            os.makedirs(os.path.join(output_folder, f"{file_name}"))
        result_file_path = os.path.join(output_folder, f"{file_name}", f"{file_name}_results.npy")
        np.save(result_file_path, file_results)

def main():
    project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    audio_folder = os.path.join(project_folder, "data/audios")
    output_folder = os.path.join(project_folder, "data/audio_features")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = process_audio_files(audio_folder, num_bins=32)
    write_results(results, output_folder)

if __name__ == "__main__":
    main()
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

def approximate_by_distribution(signal, distribution_func, frame_length=1024, hop_length=512):
    if distribution_func == pareto:
        feature = amplitude_envelope(signal, frame_length, hop_length)
    elif distribution_func == skewnorm:
        feature = zero_crossing_rate(signal, frame_length, hop_length)
    else:
        raise ValueError("Unsupported distribution function")

    return distribution_func.fit(feature)

def plot_and_save_histogram(signal, distribution_func, file_path, frame_length=1024, hop_length=512, num_bins=50):
    params = approximate_by_distribution(signal, distribution_func, frame_length, hop_length)
    feature = amplitude_envelope(signal, frame_length, hop_length) if distribution_func == pareto else zero_crossing_rate(signal, frame_length, hop_length)

    plt.figure(figsize=(10, 4))
    plt.hist(feature, bins=num_bins, density=True, alpha=0.6, color='g')
    x = np.linspace(min(feature), max(feature), 1000)
    plt.plot(x, distribution_func.pdf(x, *params), 'r-', lw=2)
    plt.xlabel('Feature Value')
    plt.ylabel('Probability Density')
    plt.title(f'{"Pareto" if distribution_func == pareto else "Skew Normal"} Distribution')
    plt.savefig(file_path)
    plt.close()

def process_audio_files(audio_folder, output_folder):
    results = []
    for file in os.listdir(audio_folder):
        if file.endswith('.wav'):
            # Use os.path.splitext to correctly handle file names with multiple periods
            file_name_without_extension = os.path.splitext(file)[0]

            # Create a unique directory for each audio file's results
            file_results_folder = os.path.join(output_folder, file_name_without_extension)
            if not os.path.exists(file_results_folder):
                os.makedirs(file_results_folder)

            file_path = os.path.join(audio_folder, file)
            signal, sr = load_audio(file_path)

            if signal is not None:
                skew_params = approximate_by_distribution(signal, skewnorm)
                pareto_params = approximate_by_distribution(signal, pareto)

                combined_params = np.hstack([pareto_params, skew_params])
                results.append((file_name_without_extension, combined_params))

                # Update file paths to include the new subdirectories
                plot_and_save_histogram(signal, skewnorm, os.path.join(file_results_folder, f"{file_name_without_extension}_audio_skew.png"))
                plot_and_save_histogram(signal, pareto, os.path.join(file_results_folder, f"{file_name_without_extension}_audio_pareto.png"))

    return results

def write_results(results, output_folder):
    for file_name, file_results in results:
        np.save(os.path.join(output_folder, file_name, f"{file_name}_results.npy"), file_results)

def main():
    project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    audio_folder = os.path.join(project_folder, "data/audios")
    output_folder = os.path.join(project_folder, "data/audio_features")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = process_audio_files(audio_folder, output_folder)
    write_results(results, output_folder)

if __name__ == "__main__":
    main()
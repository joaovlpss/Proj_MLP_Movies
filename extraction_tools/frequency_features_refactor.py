import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

def calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate):
    """
    Calculate the bin index in the spectrogram that corresponds to a given split frequency.

    Parameters:
    spectrogram (numpy.ndarray): The spectrogram of the audio signal.
    split_frequency (float): The frequency at which to split the spectrogram.
    sample_rate (int): The sample rate of the audio signal.

    Returns:
    int: The bin index corresponding to the split frequency.
    """
    frequency_range = sample_rate / 2
    frequency_delta = frequency_range / spectrogram.shape[0]
    split_frequency_bin = np.floor(split_frequency / frequency_delta)
    return int(split_frequency_bin)

def calculate_band_energy_ratio(spectrogram, split_frequency, sample_rate):
    """
    Calculate the Band Energy Ratio for each frame of the spectrogram.

    Parameters:
    spectrogram (numpy.ndarray): The spectrogram of the audio signal.
    split_frequency (float): The frequency used to split the spectrogram into low and high bands.
    sample_rate (int): The sample rate of the audio signal.

    Returns:
    numpy.ndarray: An array of band energy ratio values for each frame.
    """
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

def process_audio_file(file_path):
    """
    Process an audio file to calculate the Band Energy Ratio and fit the Spectral Centroid to a Skew Normal distribution.

    Parameters:
    file_path (str): The path to the audio file.

    Returns:
    tuple: A tuple containing the energy ratio, skew parameters, and spectral centroid.
    """
    signal, sr = librosa.load(file_path)
    spectrogram = librosa.stft(signal, n_fft=2048, hop_length=512)

    # Calculate Band Energy Ratio
    band_energy_ratio = calculate_band_energy_ratio(spectrogram, 2000, sr)
    energy_ratio = np.sum(band_energy_ratio > 10000) / np.sum(band_energy_ratio <= 10000)

    # Calculate and fit Spectral Centroid to Skew Normal Distribution
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    skew_params = skewnorm.fit(spectral_centroid)

    return energy_ratio, skew_params, spectral_centroid

def save_histogram(centroid_values, skew_params, output_path, num_bins=50):
    """
    Save a histogram of spectral centroid values with a skew normal distribution fit.

    Parameters:
    centroid_values (numpy.ndarray): Array of spectral centroid values.
    skew_params (tuple): Parameters of the fitted skew normal distribution.
    output_path (str): Path where the histogram will be saved.
    num_bins (int): Number of bins for the histogram.
    """
    plt.figure(figsize=(10, 4))
    plt.hist(centroid_values, bins=num_bins, density=True, alpha=0.6, color='g')
    x = np.linspace(min(centroid_values), max(centroid_values), 1000)
    plt.plot(x, skewnorm.pdf(x, *skew_params), 'r-', lw=2)
    plt.xlabel('Spectral Centroid')
    plt.ylabel('Probability Density')
    plt.title('Spectral Centroid - Skew Normal Fit')
    plt.savefig(output_path)
    plt.close()

def main():
    """
    Main function to process all audio files in a specified directory,
    calculating their band energy ratio and spectral centroid, and saving the results.
    """
    project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    audio_folder = os.path.join(project_folder, "data/audios")
    output_folder = os.path.join(project_folder, "data/frequency_features")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(audio_folder):
        if file_name.endswith('.wav'):
            # Create a unique directory for each audio file's results
            file_name_without_extension = os.path.splitext(file_name)[0]
            file_results_folder = os.path.join(output_folder, file_name_without_extension)
            if not os.path.exists(file_results_folder):
                os.makedirs(file_results_folder)

            file_path = os.path.join(audio_folder, file_name)
            energy_ratio, skew_params, spectral_centroid = process_audio_file(file_path)

            # Save histogram
            hist_output_path = os.path.join(file_results_folder, f"frequency_{file_name_without_extension}_hist.png")
            save_histogram(spectral_centroid, skew_params, hist_output_path)

            # Save results as .npy file
            np.save(os.path.join(file_results_folder, "frequency_analysis_results.npy"), np.array([energy_ratio, *skew_params]))

if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import librosa

def open_histogram():
    file_path = filedialog.askopenfilename(title="Open Histogram File", filetypes=[("Numpy Files", "*.npy")])
    if file_path:
        histogram = np.load(file_path)

        # Assuming the histogram is evenly split into H, S, and V components
        split_length = len(histogram) // 3
        hist_h = histogram[:split_length]
        hist_s = histogram[split_length:2 * split_length]
        hist_v = histogram[2 * split_length:]

        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 6))

        axs[0].bar(range(len(hist_h)), hist_h, color='r')
        axs[0].set_title("Hue (H) Channel")

        axs[1].bar(range(len(hist_s)), hist_s, color='g')
        axs[1].set_title("Saturation (S) Channel")

        axs[2].bar(range(len(hist_v)), hist_v, color='b')
        axs[2].set_title("Value (V) Channel")

        plt.tight_layout()
        plt.show()


def open_haralick():
    file_path = filedialog.askopenfilename(title="Open Haralick Feature File", filetypes=[("Numpy Files", "*.npy")])
    if file_path:
        haralick_features = np.load(file_path)
        feature_names = ["Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance", 
                         "Inverse Difference Moment", "Sum Average", "Sum Variance", "Sum Entropy", 
                         "Entropy", "Difference Variance", "Difference Entropy", "Information Measures of Correlation 1",
                         "Information Measures of Correlation 2", "Maximal Correlation Coefficient"]
        feature_info = "\n".join([f"{name}: {value:.4f}" for name, value in zip(feature_names, haralick_features)])
        messagebox.showinfo("Haralick Features", feature_info)

        # Optional: Display a plot
        plt.figure()
        plt.bar(range(len(haralick_features)), haralick_features)
        plt.xticks(range(len(haralick_features)), labels=feature_names, rotation=90)
        plt.title("Haralick Features")
        plt.show()

def open_audio_features():
    file_path = filedialog.askopenfilename(title="Open Audio Feature File", filetypes=[("Numpy Files", "*.npy")])
    if file_path:
        audio_features = np.load(file_path)

        # Plotting the audio features
        plt.figure(figsize=(10, 4))
        plt.plot(audio_features)
        plt.title("Audio Amplitude Envelope")
        plt.xlabel("Frames")
        plt.ylabel("Amplitude")
        plt.show()

root = tk.Tk()
root.title("Feature Visualization Tool")

btn_open_histogram = tk.Button(root, text="Open HSV Histogram", command=open_histogram)
btn_open_histogram.pack(pady=10)

btn_open_haralick = tk.Button(root, text="Open Haralick Feature", command=open_haralick)
btn_open_haralick.pack(pady=10)

btn_open_audio_features = tk.Button(root, text="Open Audio Features", command=open_audio_features)
btn_open_audio_features.pack(pady=10)

root.mainloop()

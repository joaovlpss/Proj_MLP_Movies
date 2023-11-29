import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
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

def label_features(features, script_number):
    labeled_features = {}
    if script_number == 1:
        # Assuming the first three values are Pareto parameters and the next three are Skewnorm parameters
        labeled_features['Pareto Params'] = features[:3]
        labeled_features['Skewnorm Params'] = features[3:6]
    elif script_number == 2:
        # The first value is the Band Energy Ratio and the next three are Skew Params
        labeled_features['Band Energy Ratio'] = features[0]
        labeled_features['Skew Params'] = features[1:4]

    return labeled_features

def open_audio_features():
    file_path = filedialog.askopenfilename(title="Open Audio Feature File", filetypes=[("Numpy Files", "*.npy")])
    if file_path:
        audio_features = np.load(file_path)
        # Determine which script the file is from based on the length of the loaded array
        script_number = 1 if len(audio_features) == 6 else 2
        labeled_features = label_features(audio_features, script_number)
        messagebox.showinfo("Audio Features", '\n'.join([f"{key}: {value}" for key, value in labeled_features.items()]))

def open_histogram_image():
    file_path = filedialog.askopenfilename(title="Open Histogram Image", filetypes=[("Image Files", "*.png")])
    if file_path:
        img = mpimg.imread(file_path)
        plt.imshow(img)
        plt.axis('off')  # No axes for images
        plt.show()

def open_combined_feature_vector():
    file_path = filedialog.askopenfilename(title="Open Combined Feature Vector", filetypes=[("Numpy Files", "*.npy")])
    if file_path:
        combined_features = np.load(file_path)

        # Create a new window to display the feature vector
        feature_window = tk.Toplevel(root)
        feature_window.title("Combined Feature Vector")

        # Create a scrolled text widget to display the data
        text_area = scrolledtext.ScrolledText(feature_window, wrap=tk.WORD, width=40, height=10)
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Convert the numpy array to a string and insert into the text widget
        feature_text = '\n'.join([f"{i}: {value:.10f}" for i, value in enumerate(combined_features)])
        text_area.insert(tk.INSERT, feature_text)
        text_area.config(state=tk.DISABLED)  # Make the text read-only

root = tk.Tk()
root.title("Feature Visualization Tool")

btn_open_histogram = tk.Button(root, text="Open HSV Histogram", command=open_histogram)
btn_open_histogram.pack(pady=10)

btn_open_haralick = tk.Button(root, text="Open Haralick Feature", command=open_haralick)
btn_open_haralick.pack(pady=10)

btn_open_audio_features = tk.Button(root, text="Open Audio Features", command=open_audio_features)
btn_open_audio_features.pack(pady=10)

btn_open_histogram_image = tk.Button(root, text="Open Histogram Image", command=open_histogram_image)
btn_open_histogram_image.pack(pady=10)

btn_open_combined_feature_vector = tk.Button(root, text="Open Combined Feature Vector", command=open_combined_feature_vector)
btn_open_combined_feature_vector.pack(pady=10)

root.mainloop()

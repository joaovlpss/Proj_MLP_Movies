import os
from moviepy.editor import VideoFileClip

def extract_audio_from_video(video_path, audio_path):
    """
    Extract the audio from a video file and save it as a .wav file.
    :param video_path: Path to the video file.
    :param audio_path: Path to save the extracted audio file.
    """
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

def main():
    # Define the folder paths
    parent_folder = os.path.dirname(os.path.abspath(__file__))
    movies_folder = os.path.join(parent_folder, "../data/movies")
    audios_folder = os.path.join(parent_folder, "../data/audios")

    # Ensure audios folder exists
    if not os.path.exists(audios_folder):
        os.makedirs(audios_folder)

    # Supported video formats
    supported_formats = ('.mkv', '.avi', '.mp4')

    # Process each video file
    for file in os.listdir(movies_folder):
        if (file.endswith(supported_formats)):
            video_path = os.path.join(movies_folder, file)
            # Replace the last occurrence of a period in the filename with .wav
            audio_file_name = '.'.join(file.split('.')[:-1]) + '.wav'
            audio_path = os.path.join(audios_folder, audio_file_name)

            extract_audio_from_video(video_path, audio_path)
            print(f"Extracted audio from {file} to {audio_file_name}")

if __name__ == "__main__":
    main()

import os
import re
import shutil

def organize_movies(movies_dir):
    for genre in os.listdir(movies_dir):
        genre_path = os.path.join(movies_dir, genre)
        if os.path.isdir(genre_path):
            for movie_folder in os.listdir(genre_path):
                movie_folder_path = os.path.join(genre_path, movie_folder)
                if os.path.isdir(movie_folder_path):
                    for file in os.listdir(movie_folder_path):
                        if file.endswith(('.mp4', '.mkv', '.avi')):
                            new_name = rename_movie(file, genre)
                            old_file_path = os.path.join(movie_folder_path, file)
                            new_file_path = os.path.join(movies_dir, new_name)
                            shutil.move(old_file_path, new_file_path)
                            print(f"Moved: {new_file_path}")

def rename_movie(filename, genre):
    # Extract name and year from the filename
    match = re.match(r"(.+?)\.(\d{4})\..*\.(mp4|mkv|avi)$", filename)
    if match:
        name, year, ext = match.groups()
        new_name = f"{genre}_{name}.{year}.{ext}"
        return new_name
    else:
        return filename

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    organize_movies(current_dir)

import cv2
import os


path1 = "./PetImages/Cat"
path2 = "./PetImages/Dog"

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])

def delete_corrupted_or_none_files(folder):
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if is_image_file(filepath):
            try:
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if img is None:
                    os.remove(filepath)
                    print(f"Deleted {filepath} because it is None")
            except cv2.error as e:
                os.remove(filepath)
                print(f"Deleted {filepath} due to OpenCV error: {e}")
            except Exception as e:
                os.remove(filepath)
                print(f"Error processing {filepath}: {e}")

delete_corrupted_or_none_files(path1)
delete_corrupted_or_none_files(path2)

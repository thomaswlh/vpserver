import cv2
from ultralytics import YOLO
import torch

def read_images(image_paths):
    """
    Reads images from a list of file paths and returns them as a list.
    Args:
        image_paths (list of str): List of image file paths.
    Returns:
        list: List of images read by cv2.imread (may contain None for unreadable files).
    """
    images = []
    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Unable to read image at '{path}'.")
            images.append(img)
        except Exception as e:
            print(f"Error reading image at '{path}': {e}")
            images.append(None)
    return images

def load_segmentation_model(model_path='yolo11n-seg.pt'):
    """
    Loads the YOLO segmentation model and moves it to GPU if available.
    Args:
        model_path (str): Path to the YOLO segmentation model weights.
    Returns:
        YOLO: Loaded YOLO segmentation model, or None if loading fails.
    """
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model from '{model_path}': {e}")
        return None

    try:
        if torch.cuda.is_available():
            model.to('cuda')
        else:
            print("CUDA is not available. The model will run on CPU.")
    except Exception as e:
        print(f"Error moving model to CUDA: {e}")
        return None

    return model
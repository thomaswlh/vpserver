import cv2
from ultralytics import YOLO
import torch
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt

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

def select_roi_points(frame, window_name="Select ROI"):
    """
    Allows the user to select a ROI on the given frame using OpenCV's GUI.
    Returns:
        tuple: ((x1, y1), (x2, y2)) of the selected ROI, or None if selection is invalid.
    """
    if frame is None:
        print("Error: Input frame is None.")
        return None, None
    if not hasattr(cv2, "selectROI"):
        print("Error: cv2.selectROI is not available in your OpenCV installation.")
        return None, None
    try:
        while True:
            roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
            x, y, w, h = roi
            cv2.destroyWindow(window_name)
            if w == 0 or h == 0:
                print("Invalid ROI selected (zero width or height). Please select again.")
                continue
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            return (x1, y1), (x2, y2)
    except Exception as e:
        print(f"Error during ROI selection: {e}")
        return None, None

def select_roi_points_matplotlib(frame, window_name="Select ROI"):
    """
    Allows the user to select a ROI on the given frame using matplotlib's RectangleSelector.
    Returns:
        tuple: ((x1, y1), (x2, y2)) of the selected ROI, or (None, None) if selection is invalid.
    """
    if frame is None:
        print("Error: Input frame is None.")
        return None, None

    roi = {}

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        roi['pt1'] = (min(x1, x2), min(y1, y2))
        roi['pt2'] = (max(x1, x2), max(y1, y2))
        plt.close()

    fig, ax = plt.subplots()
    ax.imshow(frame[..., ::-1])  # Convert BGR (OpenCV) to RGB (matplotlib)
    ax.set_title(window_name)
    rect_selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1], minspanx=5, minspany=5, spancoords='pixels',
        interactive=True
    )
    plt.show()

    if 'pt1' in roi and 'pt2' in roi:
        x1, y1 = roi['pt1']
        x2, y2 = roi['pt2']
        if x1 == x2 or y1 == y2:
            print("Invalid ROI selected (zero width or height).")
            return None, None
        return (x1, y1), (x2, y2)
    else:
        print("ROI selection cancelled or invalid.")
        return None, None

def mask_out_rectangle(mask, pt1, pt2):
    """
    Draws a black rectangle on the mask image between pt1 and pt2.

    Args:
        mask (np.ndarray): The mask image (single channel or 3-channel).
        pt1 (tuple): (x1, y1) - top-left or one corner of the rectangle.
        pt2 (tuple): (x2, y2) - bottom-right or opposite corner of the rectangle.

    Returns:
        np.ndarray: The mask image with the rectangle region set to black, or None if error.
    """
    try:
        if mask is None:
            print("Error: Input mask is None.")
            return None
        if not (isinstance(pt1, tuple) and isinstance(pt2, tuple)):
            print("Error: Points must be tuples.")
            return None
        if len(pt1) != 2 or len(pt2) != 2:
            print("Error: Points must be (x, y) tuples.")
            return None
        mask_copy = mask.copy()
        cv2.rectangle(mask_copy, pt1, pt2, color=0, thickness=-1)
        return mask_copy
    except Exception as e:
        print(f"Error drawing rectangle on mask: {e}")
        return None
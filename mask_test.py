from ultralytics import YOLO
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config
import numpy as np
import torch, cv2

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")

seg_model = YOLO("yolo11n-seg.pt")
lama_model = ModelManager(name="lama", device="cuda" if torch.cuda.is_available() else "cpu")  

def combine_masks(mask_list):
    """
    Combines multiple masks into a single mask.
    Args:
        mask_list (list of numpy arrays): List of individual masks.
    Returns:
        numpy array: Combined mask.
    """
    if not mask_list:
        return None  # Return None if the list is empty

    # Combine masks by taking the maximum value at each pixel position
    combined_mask = mask_list[0]
    for mask in mask_list[1:]:
        combined_mask = np.maximum(combined_mask, mask)

    return combined_mask

def get_image_list(image_path_list):
    """
    Get a list of images from the provided paths.
    Args:
        image_path_list (list of str): List of paths to the images.
    Returns:
        list of numpy arrays: List of images.
    """
    images = []
    for image_path in image_path_list:
        # Read the image
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
        else:
            print(f"Error reading image: {image_path}")
    return images

def get_people_masks(model, images_list):
    
    # Loop through each image path
    # and predict with the segmentation model
    # and save the masks

    people_masks = []

    for image in images_list:

        img_masks = []
        
        # Predict with segmentation model
        numpy_results = model.predict(
            source=image,  # Use absolute path
            conf=0.7,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            save=True,
            verbose=False
            )[0].numpy()

        # Get object masks
        masks = numpy_results.masks.data

        for i in range(len(masks)):
            filename = f"{numpy_results.names[int(numpy_results.boxes.cls[i])]}{i}.jpg"
            if "person" in filename:
                img_masks.append(masks[i]*255)

        # Combine masks for the current image
        people_masks.append(combine_masks(img_masks))

    return people_masks

def lama_remove_people(orginal_image_path_list, mask_list):
    """
    Remove people from the original image using the provided masks.
    Args:
        orginal_image_path_list (list of str): List of paths to the original images.
        mask_list (list of numpy arrays): List of masks for each image.
    Returns:
        list of numpy arrays: List of images with people removed.
    """
    # Loop through each image path
    # and predict with the segmentation model
    # and save the masks

    results = []

    for i in range(len(orginal_image_path_list)):
        image_path = orginal_image_path_list[i]
        mask = mask_list[i]

        # Read the original image
        original_image = cv2.imread(image_path)

        # Remove people using the mask
        # result = lama_model.remove(original_image, mask)
        results.append(result)

    return results



if __name__ == "__main__":

    images_path = ["./people.jpeg", "./bus.jpg"]

    masks = get_people_masks(seg_model, get_image_list(images_path))

    for mask in masks:
        cv2.imshow(f"mask", mask)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
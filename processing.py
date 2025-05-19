import cv2, os
import numpy as np


def process_segmentation(model, images_list, images_path_list, output_mask_path=None, device="cuda:0", classes=[0], output_mask=False, show_mask=False, show_detetion=False):
    """
    Runs segmentation on a list of images and saves/displays the combined masks.

    Args:
        model: Loaded YOLO segmentation model.
        images_list (list): List of images (np.ndarray) to process.
        images_path_list: List of image paths of the images in images_list.
        output_mask_path (str or None): Path to folder to save the output mask images, or None.
        device (str): Device to run inference on.
        classes (list): List of class IDs to segment.
        output_mask (bool): Whether to save the mask to a file.
        show_mask (bool): Whether to display the mask using OpenCV.
        show_detetion (bool): Whether to display the detection results.

    Returns:
        list: List of combined masks as NumPy arrays, or None if failed.
        list: List of paths to saved mask images, or None if not saved.
    """
    result_masks = []
    output_mask_paths = []

    for idx, image in enumerate(images_list):
        if image is None:
            print(f"Warning: Image at index {idx} is None.")
            result_masks.append(None)
            continue
        try:
            results = model(source=image, device=device, classes=classes, retina_masks=True, save=True)
        except Exception as e:
            print(f"Error running model inference on image {idx}: {e}")
            result_masks.append(None)
            continue

        for result in results:
            masks = []
            try:
                if not hasattr(result, 'masks') or result.masks is None:
                    print(f"Warning: No masks found in result {idx}.")
                    result_masks.append(None)
                    continue
                for mask_idx, mask in enumerate(result.masks):
                    try:
                        mask_np = mask.data.squeeze(0).cpu().numpy().astype(np.uint8) * 255
                        masks.append(mask_np)
                    except Exception as e:
                        print(f"Error processing mask {mask_idx} in result {idx}: {e}")
                        continue
                if masks:
                    try:
                        combined_mask = np.maximum.reduce(masks)
                        result_masks.append(combined_mask)
                        if output_mask and output_mask_path is not None:
                            try:
                                # Save with index in filename if processing multiple images
                                save_path = output_mask_path
                                filename = os.path.basename(images_path_list[idx])
                                save_path = os.path.join(output_mask_path, f"{filename}_MASK.jpg")
                                print(f"Saving mask to {save_path}")
                                # Ensure the directory exists
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                # Save the mask
                                cv2.imwrite(save_path, combined_mask)
                                output_mask_paths.append(save_path)
                            except Exception as e:
                                print(f"Error saving mask to {save_path}: {e}")
                                output_mask_paths.append(None)
                        if show_mask:
                            try:
                                cv2.imshow('Mask', combined_mask)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                            except Exception as e:
                                print(f"Error displaying mask: {e}")
                    except Exception as e:
                        print(f"Error combining masks in result {idx}: {e}")
                        result_masks.append(None)
                else:
                    print(f"Warning: No valid masks to combine in result {idx}.")
                    result_masks.append(None)
            except Exception as e:
                print(f"Unexpected error in result {idx}: {e}")
                result_masks.append(None)
    return result_masks, output_mask_paths

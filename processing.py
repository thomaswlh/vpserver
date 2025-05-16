import torch, cv2
import numpy as np
from ultralytics import YOLO


def process_segmentation(model, image_path, output_mask_path, device="cuda:0", output_mask=False, show_mask=False):
    """
    Runs segmentation on an image and saves/displays the combined mask.

    Args:
        model: Loaded YOLO segmentation model.
        image_path (str): Path to the input image.
        output_mask_path (str): Path to save the output mask image.
        device (str): Device to run inference on.
        class_ids (list): List of class IDs to segment.
        output_mask (bool): Whether to save the mask to a file.
        show_mask (bool): Whether to display the mask using OpenCV.

    Returns:
        list: List of combined masks as NumPy arrays, or None if failed.
    """
    result_masks = []
    try:
        results = model(source=image_path, device=device, classes=[0], retina_masks=True)
    except Exception as e:
        print(f"Error running model inference: {e}")
        return None

    for idx, result in enumerate(results):
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
                    if output_mask:
                        try:
                            cv2.imwrite(output_mask_path, combined_mask)
                        except Exception as e:
                            print(f"Error saving mask to {output_mask_path}: {e}")
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
    return result_masks

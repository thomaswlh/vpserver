import cv2
from utils import *
from processing import *

def main():

    # Load model
    model = load_segmentation_model('yolo11n-seg.pt')
    if model is None:
        print("Failed to load model.")
        return

    # Read image
    images_path = ["/Users/thomaswu/Downloads/5月4日16：06 闲人/DJI_20250504160641_0002_V_航点6.jpg"]
    images = read_images(images_path)
    for idx, image in enumerate(images):
        if image is None:
            print(f"Failed to read image: {images_path[idx]}")

    # Process segmentation
    result_masks, output_mask_paths = process_segmentation(
        model=model,
        images_list=images,
        images_path_list=images_path,
        output_mask_path="./media/masks",
        device="cpu",
        output_mask=True,  # <-- Change this to True
        show_mask=False
    )

    # Select ROI (optional, comment out if not needed)

    for idx, mask in enumerate(result_masks):
        # Error handling: Check if output_mask_paths has enough elements
        if output_mask_paths is None or idx >= len(output_mask_paths):
            print(f"Error: No output path for mask at index {idx}. Skipping save.")
            continue

        pt1, pt2 = select_roi_points_matplotlib(mask, window_name="Select ROI")
        print(f"Selected ROI: {pt1}, {pt2}")
        processed_mask = mask_out_rectangle(mask, pt1, pt2)
        if processed_mask is not None:
            success = cv2.imwrite(output_mask_paths[idx], processed_mask)
            if success:
                print(f"Segmentation and mask saving for {output_mask_paths[idx]} successful.")
            else:
                print(f"Failed to save processed mask to {output_mask_paths[idx]}.")
        else:
            print(f"Segmentation for {output_mask_paths[idx]} failed.")

if __name__ == "__main__":
    main()
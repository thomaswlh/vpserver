import cv2
from utils import *
from processing import *

def main():
    image_path = './media/DJI_20250504171801_0005_V.JPG'
    output_mask_path = './media/DJI_20250504171801_0005_V_MASK.JPG'

    # Load model
    model = load_segmentation_model('yolo11n-seg.pt')
    if model is None:
        print("Failed to load model.")
        return

    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to read image: {image_path}")
        return

    # Process segmentation
    result_masks = process_segmentation(
        model=model,
        image_path=image_path,
        output_mask_path=None,
        device="cuda:0",
        output_mask=False,
        show_mask=False
    )

    # Select ROI (optional, comment out if not needed)
    pt1, pt2 = select_roi_points(result_masks[0])
    print(f"Selected ROI: {pt1}, {pt2}")

    processed_mask = mask_out_rectangle(result_masks[0], pt1, pt2)

    cv2.imwrite(output_mask_path, processed_mask)

    if result_masks is not None and result_masks[0] is not None:
        print("Segmentation and mask saving successful.")
    else:
        print("Segmentation failed.")

if __name__ == "__main__":
    main()
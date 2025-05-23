import torch, cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv11-seg model (replace with the correct path/model name if needed)
model = YOLO('./weights/yolo11n-seg.pt')

# Ensure CUDA is available and use GPU
if torch.cuda.is_available():
    model.to('cuda')
else:
    print("CUDA is not available. The model will run on CPU.")

# Run segmentation inference on an image
results = model(source='/Users/thomaswu/Downloads/5月4日16：06 闲人/DJI_20250504160641_0002_V_航点6.jpg',device="cpu",classes=[0], retina_masks=True, conf=0.3) # Replace 'input.jpg' with your image path

result_masks = []
masks = []

for result in results:
    masks = []
    for mask in result.masks:
        print(mask.shape)
        # Convert mask to numpy, ensure uint8 and binary (0 or 255)
        mask_np = mask.data.squeeze(0).cpu().numpy().astype(np.uint8) * 255  # shape: (H, W)
        masks.append(mask_np)
    result_masks.append(np.maximum.reduce(masks))  # Combine masks if multiple exist
    cv2.imshow('Mask', result_masks[-1])  # Display the mask
    cv2.imwrite('./media/DJI_20250504171801_0005_V_MASK.JPG', result_masks[-1])
    cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

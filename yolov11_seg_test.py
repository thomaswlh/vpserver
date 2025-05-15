import torch, cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv11-seg model (replace with the correct path/model name if needed)
model = YOLO('yolo11n-seg.pt')

# Ensure CUDA is available and use GPU
if torch.cuda.is_available():
    model.to('cuda')
else:
    print("CUDA is not available. The model will run on CPU.")

# Run segmentation inference on an image
results = model(source='DJI_20250504171801_0005_V.JPG',device="cuda:0",classes=[0], retina_masks=True) # Replace 'input.jpg' with your image path

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
    cv2.imwrite('DJI_20250504171801_0005_V_MASK.JPG', result_masks[-1])
    cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import einops

from ControlNet.annotator.util import resize_image, HWC3
from ControlNet.annotator.openpose import OpenposeDetector

class IdentityPoseDataset(Dataset):
    def __init__(self, image_dir, device, transform=None, image_height=512, image_width=512, image_resolution=512):
        self.image_paths = list(Path(image_dir).glob('*.jpg'))
        self.device = device
        self.transform = transform

        self.image_height = image_height
        self.image_width = image_width
        self.image_resolution = image_resolution

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.apply_openpose = OpenposeDetector()
        
    def __len__(self):
        return len(self.image_paths)

    def _get_pose_from_image(self, image):
        condition_image = HWC3(image)
        detected_map, _ = self.apply_openpose(resize_image(condition_image(self.image_height)))
        detected_map = HWC3(detected_map)
        
        detected_map = cv2.resize(detected_map, (512, 512), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
        num_samples = 1 # only using one pose
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        control_cond = {"c_concat": [control], "c_crossattn": [] }
        control_uncond = {"c_concat": [control], "c_crossattn": []}

        return detected_map, control_cond, control_uncond

    def _get_face_from_image(self, image, padding=50):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None
        
        # Get the first face (should be single object)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image.shape[1])
        y2 = min(y + h + padding, image.shape[0])

        face = image[y1:y2, x1:x2]

        return face

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        detected_map, pose_condition, pose_uncond = self._get_pose_from_image(image) 
        identity_condition = self._get_face_from_image(image)    

        return image, detected_map, pose_condition, pose_uncond, identity_condition

# Preview some samples
if __name__ == "__main__":
    ident_dataset = IdentityPoseDataset('./coco2017/test', 'cpu')

    for ind in range(5):
        image, pose_map, _, _, cropped_face = ident_dataset[ind]

        image.show()
        pose_map.show()
        cropped_face.show()

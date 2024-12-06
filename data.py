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

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.apply_openpose = OpenposeDetector()
        
    def __len__(self):
        return len(self.image_paths)

    def _get_pose_from_image(self, image):
        condition_image = HWC3(np.array(image))
        detected_map, _ = self.apply_openpose(resize_image(condition_image, self.image_height))
        detected_map = HWC3(detected_map)
        
        detected_map = cv2.resize(detected_map, (512, 512), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().to(self.device) / 255.0
        num_samples = 1 # only using one pose
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        control_cond = {"c_concat": [control], "c_crossattn": [] }
        control_uncond = {"c_concat": [control], "c_crossattn": []}

        return detected_map, control_cond, control_uncond

    def _get_face_from_image(self, image, padding=50):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = self.face_cascade.detectMultiScale(
            image, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) != 1:
            return None
       
        x, y, w, h = faces[0]

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
        identity_condition = self._get_face_from_image(np.array(image))    

        return image, detected_map, pose_condition, pose_uncond, identity_condition

class PreppedIdentityPoseDataset(Dataset):
    def __init__(self, ref_dir, pose_dir, ident_dir, prompt_path, device):
        self.ref_paths = list(Path(ref_dir).glob('*.jpg'))
        self.pose_paths = list(Path(pose_dir).glob('*.jpg'))
        self.ident_paths = list(Path(ident_dir).glob('*.jpg'))
        with open(prompt_path, 'r') as f:
            self.prompts = [line.strip() for line in f]

        assert(len(self.ref_paths) == len(self.pose_paths) == len(self.ident_paths))

        self.device = device

    def __len__(self):
        return len(self.ref_paths)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        ref = np.array(Image.open(self.ref_paths[idx]))
        pose = np.array(Image.open(self.pose_paths[idx]))
        ident = np.array(Image.open(self.ident_paths[idx]))
        
        control = torch.from_numpy(pose.copy()).float().to(self.device) / 255.0
        num_samples = 1 # only using single object
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        control_cond = {"c_concat": [control], "c_crossattn": []}
        control_uncond = {"c_concat": [control], "c_crossattn": []}

        return prompt, torch.Tensor(ref), control_cond, control_uncond, torch.Tensor(ident)


# Preview some samples
def prep_data():
    from tqdm import tqdm

    ident_dataset = IdentityPoseDataset('./val2017', 'cpu')

    print(len(ident_dataset))

    def array_to_pil(array):

        return Image.fromarray(np.uint8(array))

    num_single_idents = 0
    for ind in tqdm(range(len(ident_dataset))):
        try:
            image, pose_map, _, _, cropped_face = ident_dataset[ind]

            if cropped_face is not None:
                num_single_idents += 1
                pose_image = array_to_pil(pose_map)
                ident_image = array_to_pil(cropped_face)

                image.save(f'./prepped_idents/refs/ref_image_{ind}.jpg')
                pose_image.save(f'./prepped_idents/poses/pose_image_{ind}.jpg')
                ident_image.save(f'./prepped_idents/idents/ident_image_{ind}.jpg')
        except:
            pass

if __name__ == "__main__":
    prepped_dataset = PreppedIdentityPoseDataset("./prepped_idents/refs", "./prepped_idents/poses", "./prepped_idents/idents", 'cpu')
    print(f"{len(prepped_dataset)=}")

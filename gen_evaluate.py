
from evaluation.single_object.data import get_combinations
from demo.pipeline import convert_model_to_pipeline
from fastcomposer.utils import parse_args
from accelerate.utils import set_seed
from accelerate import Accelerator
import torch
import glob 
import PIL
import os 

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import types
import cv2 
import einops
from typing import Any, Callable, Dict, List, Optional, Union

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import PNDMScheduler
from transformers import CLIPTokenizer

from fastcomposer.transforms import get_object_transforms
from fastcomposer.data import DemoDataset
from fastcomposer.model import FastComposerModel
from fastcomposer.pipeline import stable_diffusion_call_with_references_delayed_conditioning
from fastcomposer.utils import parse_args

from ControlNet.annotator.util import resize_image, HWC3
from ControlNet.annotator.openpose import OpenposeDetector
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler
from ControlNet.ldm.modules.diffusionmodules.util import make_ddim_timesteps, make_ddim_sampling_parameters, noise_like, extract_into_tensor

from knit import CombinedSampler
from knit import apply_openpose

@torch.no_grad()
def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Setup weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    cn_model_path = "./ControlNet/models/cldm_v15.yaml"
    cn_state_dict_path = "./ControlNet/models/control_sd15_openpose.pth"

    # Initialize combined sampler
    sampler = CombinedSampler(cn_model_path, cn_state_dict_path)

    # Setup both components
    sampler.setup_fastcomposer(args, accelerator, weight_dtype)

    unique_token = "<|image|>"

    shape = (1, 4, 512, 512)  # Your desired shape

    strength = 1.0
    sampler.control_model.control_scales = [strength] * 13

    os.makedirs(args.output_dir, exist_ok=True)

    prompt_subject_pairs = get_combinations(
        unique_token, is_fastcomposer=True, split="eval"
    )

    reference_folder = args.test_reference_folder

    for case_id, (prompt_list, subject) in enumerate(prompt_subject_pairs):
        print(case_id, subject)
        real_case_id = case_id + args.start_idx

        reference_image_folder = os.path.join(reference_folder, subject)

        for prompt_id, prompt in enumerate(prompt_list):
            print(prompt_id, prompt)

            condition_image_path = os.path.join(args.poses, str(prompt_id)+".png")
            condition_image = HWC3(np.array(Image.open(condition_image_path).convert("RGB")))
            detected_map, _ = apply_openpose(resize_image(condition_image, 512))
            detected_map = HWC3(detected_map)

            detected_map = cv2.resize(detected_map, (512, 512), interpolation=cv2.INTER_NEAREST)

            num_samples = 1
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            prompt_text_only = prompt.replace(unique_token, "")
            controlnet_cond = {"c_concat": [control], "c_crossattn": [sampler.control_model.get_learned_conditioning([prompt_text_only])]}
            n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
            controlnet_un_cond = {"c_concat": [control], "c_crossattn": [sampler.control_model.get_learned_conditioning([n_prompt])]}

            args.test_caption = prompt
            args.test_reference_folder = reference_image_folder

            images = sampler.combined_sampling(
                cond=None,  # Will be set by FastComposer processing
                controlnet_cond=controlnet_cond,
                controlnet_un_cond=controlnet_un_cond,
                shape=shape,
                unconditional_guidance_scale=args.guidance_scale,
                fastcomposer_args=args
            )

            for instance_id in range(args.num_images_per_prompt):
                images[instance_id].save(
                    os.path.join(
                        args.output_dir,
                        f"subject_{real_case_id:04d}_prompt_{prompt_id:04d}_instance_{instance_id:04d}.jpg",
                    )
                )
        


if __name__ == "__main__":
    main()

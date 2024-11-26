import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import types
import os
import cv2 
import einops
from typing import Any, Callable, Dict, List, Optional, Union

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
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

apply_openpose = OpenposeDetector()

@torch.no_grad()
def stable_diffusion_call_control_and_fastcomposer(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    channels: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt = Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    start_merge_step = 0,
    control_net_model: Optional[DDIMSampler] = None,
    control_net_cond_embed: Optional[torch.FloatTensor] = None
):
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height, 
        width,
        callback_steps, 
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    assert do_classifier_free_guidance

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
        prompt, 
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_text_only], dim=0)

    # 4. Create schedule based on the DDIMSampler
    #control_net_model.make_schedule(ddim_num_steps=num_inference_steps, ddim_eta=eta, verbose=False) 
    self.scheduler.set_timesteps(num_inferenc_steps, device=device)
    timesteps = self.scheduler.timesteps
    #control_net_timesteps = control_net_model.ddim_timesteps

    # 5. Prepare initial latents 
    latents = torch.randn(
        (1, self.unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=device
    )

    # 6. Arrange conditional embeddings
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    (
        null_prompt_embeds,
        augmented_prompt_embeds,
        text_prompt_embeds,
    ) = prompt_embeds.chunk(3)

    text_pose_embeddings = text_prompt_embeds + control_net_cond_embed
    full_embeddings = text_prompt_embeds + control_net_cond_embed + augmented_prompt_embeds
    
    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if i <= start_merge_step:
                current_prompt_embeds = torch.cat(
                    [null_prompt_embeds, text_pose_embeddings], dim=0
                )
            else: 
                current_prompt_embeds = torch.cat(
                    [null_prompt_embeds, full_embeddings], dim=0
                )

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=current_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # perform fastcomposer guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                assert 0, "Not Implemented"

            # compute the previous noise sample x_t -> x_{t-1}
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1  or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if output_type == "latent":
        images = latents
        has_nsfw_concept = None
    elif output_type == "pil":
        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker 
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, prompt_embeds.dtype
        )

        # 10. Convert to PIL
        image = self.numpy_to_pil(image)
    else:
        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, prompt_embeds.dtype
        )

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(
        images=image, nsfw_content_detected=has_nsfw_concept
    )


class CombinedSampler:
    def __init__(self, control_model_path, control_state_dict_path, schedule="linear", **kwargs):
        self.control_model = create_model(control_model_path)
        self.control_model.load_state_dict(load_state_dict(control_state_dict_path, location='cuda'))
        self.control_model = self.control_model.cuda()
        self.ddim_sampler = DDIMSampler(self.control_model)
        self.schedule = schedule
                                                            
    def setup_fastcomposer(self, args, accelerator, weight_dtype):
        """Initialize FastComposer pipeline"""
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, 
            torch_dtype=weight_dtype
        )

        model = FastComposerModel.from_pretrained(args)
        model.load_state_dict(
            torch.load(Path(args.finetuned_model_path) / "pytorch_model.bin", map_location="cpu")
        )
        model = model.to(device=accelerator.device, dtype=weight_dtype)                                                                                                                            
        pipe.unet = model.unet
        if args.enable_xformers_memory_efficient_attention:
            pipe.unet.enable_xformers_memory_efficient_attention()
        
        pipe.text_encoder = model.text_encoder
        pipe.image_encoder = model.image_encoder
        pipe.postfuse_module = model.postfuse_module
        pipe.inference = types.MethodType(
            stable_diffusion_call_control_and_fastcomposer,
            pipe
        )

        del model
        
        self.fastcomposer_pipe = pipe.to(accelerator.device)
        return pipe

    @torch.no_grad()
    def combined_sampling(
        self,
        cond,
        controlnet_cond,
        shape,
        timesteps=None,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.,
        noise_dropout=0.,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        ucg_schedule=None,
        fastcomposer_args=None,
        device='cuda'
    ):
        """Combined sampling using both FastComposer and ControlNet"""
        # Initialize noise
        if x_T is None:
            img = torch.randn(shape, device=device) 
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddim_sampler.ddpm_num_timesteps 

        # Setup FastComposer condition
        if fastcomposer_args is not None:
            tokenizer = CLIPTokenizer.from_pretrained(
                fastcomposer_args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=fastcomposer_args.revision,
            )
            object_transforms = get_object_transforms(fastcomposer_args)
            demo_dataset = DemoDataset(
                test_caption=fastcomposer_args.test_caption,
                test_reference_folder=fastcomposer_args.test_reference_folder,
                tokenizer=tokenizer,
                object_transforms=object_transforms,
                device=device,
                max_num_objects=fastcomposer_args.max_num_objects,
            )
            image_ids = os.listdir(fastcomposer_args.test_reference_folder)
            print(f"Image IDs: {image_ids}")
            demo_dataset.set_image_ids(image_ids)

            batch = demo_dataset.get_data()
            
            # Process FastComposer inputs
            input_ids = batch["input_ids"].to(device)
            image_token_mask = batch["image_token_mask"].to(device)
            all_object_pixel_values = batch["object_pixel_values"].unsqueeze(0).to(device)
            num_objects = batch["num_objects"].unsqueeze(0).to(device)
            
            # Get FastComposer embeddings
            if self.fastcomposer_pipe.image_encoder is not None:
                object_embeds = self.fastcomposer_pipe.image_encoder(all_object_pixel_values.half())
            else:
                object_embeds = None
            
            encoder_hidden_states = self.fastcomposer_pipe.text_encoder(
                input_ids, image_token_mask, object_embeds, num_objects
            )[0]

            encoder_hidden_states_text_only = pipe._encode_prompt(
                prompt_text_only,
                device,
                fastcomposer_args.num_images_per_prompt,
                do_classifier_free_guidance=False,
            )
            
            # Process combined conditioning
            cond = self.fastcomposer_pipe.postfuse_module(
                encoder_hidden_states,
                object_embeds,
                image_token_mask,
                num_objects,
            )

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) 
        total_steps = timesteps 
        
        images = pipe.inference(
            prompt_text_only,
            shape[0],
            shape[1],
            shape[2],
            total_steps,
            fastcomposer_args.num_images_per_prompt,
            fastcomposer_args.eta,
            fastcomposer_args.generator,
            control_net_model = self.ddim_sampler,
            control_net_cond_embed = control_net_cond,
            start_merge_step = 5
        )

        return images['images']


@torch.no_grad()
def main():
    # Parse arguments
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

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

    # Prepare your conditions here
    condition_image_path = "./data/celeba_test_single/001603/000000034.jpg"
    condition_image = HWC3(np.array(Image.open(condition_image_path)))
    detected_map, _ = apply_openpose(resize_image(condition_image, 512))
    detected_map = HWC3(detected_map)
    image_resolution = 512
    condition_image = resize_image(condition_image, image_resolution)
    H, W, C = condition_image.shape
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

    num_samples = 1
    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    controlnet_cond = control  # Your ControlNet condition
    shape = (1, 4, 64, 64)  # Your desired shape

    # Run combined sampling
    images, intermediates = sampler.combined_sampling(
        cond=None,  # Will be set by FastComposer processing
        controlnet_cond=controlnet_cond,
        shape=shape,
        unconditional_guidance_scale=args.guidance_scale,
        fastcomposer_args=args
    )

    # Save results
    for idx, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image = Image.fromarray(image.cpu().numpy())
        image.save(os.path.join(args.output_dir, f"combined_output_{idx}.png"))

if __name__ == "__main__":
    main()

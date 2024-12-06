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
from ControlNet.cldm.cldm import ControlLDM
from ControlNet.cldm.ddim_hacked import DDIMSampler
from ControlNet.ldm.modules.diffusionmodules.util import make_ddim_timesteps, make_ddim_sampling_parameters, noise_like, extract_into_tensor

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
    num_images_per_prompt : Optional[int] = 1,
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
    controlnet_model: Optional[ControlLDM] = None,
    controlnet_cond = None,
    controlnet_uncond = None,
    control_scales = [1.0] * 13
):
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    assert do_classifier_free_guidance

    # 2. Encode input prompt
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

    # Create schedule
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 4. Prepare initial latents 
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        self.unet.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Arrange conditional embeddings
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    (
        null_prompt_embeds,
        augmented_prompt_embeds,
        text_prompt_embeds,
    ) = prompt_embeds.chunk(3)


    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).half()

            if i <= start_merge_step:
                current_prompt_embeds = torch.cat(
                    [null_prompt_embeds, text_prompt_embeds], dim=0
                )
            else: 
                current_prompt_embeds = torch.cat(
                    [null_prompt_embeds, augmented_prompt_embeds], dim=0
                )

            control_t = torch.full((batch_size,), t, device=device, dtype=torch.long)
            # Get the controlNet preds
            all_controls = controlnet_model(
                x=latent_model_input[1:].float(),
                hint=torch.cat(controlnet_cond['c_concat'], 1),
                timesteps=control_t,
                context=torch.cat(controlnet_cond['c_crossattn'], 1),
            )
            all_controls = [c * scale for c, scale in zip(all_controls, control_scales)]

            mid_resid = all_controls.pop()
            control_range = len(all_controls)
            down_resids = reversed([all_controls.pop() for i in range(control_range)])

            mid_resid = mid_resid.half()
            down_resids = [d.half() for d in down_resids]

            # predict the noise residual from fastcomposer
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=current_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_resids,
                mid_block_additional_residual=mid_resid
            ).sample
                
            # perform fastcomposer guidance
            if do_classifier_free_guidance:
                fc_uncond, fc_text = noise_pred.chunk(2)
                #mixed_uncond = (model_uncond + fc_uncond)/2.0
                #mixed_cond = (model_output + fc_text)/2.0
                noise_pred = fc_uncond + guidance_scale * (
                    fc_text - fc_uncond 
                )
            else:
                assert 0, "Not Implemented"

            #print(f"{noise_pred.shape=}, {t=}, {latents.shape=}, {extra_step_kwargs=}")
            #print(self.scheduler)
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
        image = self.decode_latents(latents.half())

        # 9. Run safety checker 
        #image, has_nsfw_concept = self.run_safety_checker(
        #    image, device, prompt_embeds.dtype
        #)

        # 10. Convert to PIL
        image = self.numpy_to_pil(image)
    else:
        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        #image, has_nsfw_concept = self.run_safety_checker(
        #    image, device, prompt_embeds.dtype
        #)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    has_nsfw_concept = False
    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(
        images=image, nsfw_content_detected=has_nsfw_concept
    )


class CombinedSampler:
    def __init__(self, control_model_path, control_state_dict_path, schedule="linear", **kwargs):
        full_control_model = create_model(control_model_path)
        full_control_model.load_state_dict(load_state_dict(control_state_dict_path, location='cpu'))
        self.control_condition_model = full_control_model.control_model
        self.control_condition_model = self.control_condition_model.cuda()
        self.cond_stage_model = full_control_model.cond_stage_model.cuda()
        self.schedule = schedule

        sampler = DDIMSampler(full_control_model)
        self.base_timesteps = sampler.ddpm_num_timesteps

    def get_learned_conditioning(self, c):
        return self.cond_stage_model.encode(c)

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
        print(f"\n\n\n{model.unet=}\n\n\n")
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
        controlnet_un_cond,
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
            timesteps = self.base_timesteps

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
            print(f"\n\n\n{batch=}\n\n\n")
            
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

            unique_token = "<|image|>"

            prompt = fastcomposer_args.test_caption
            prompt_text_only = prompt.replace(unique_token, "")

            encoder_hidden_states_text_only = self.fastcomposer_pipe._encode_prompt(
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
        total_steps = fastcomposer_args.inference_steps 
        
        images = self.fastcomposer_pipe.inference(
            prompt_text_only,
            shape[2],
            shape[3],
            shape[1],
            total_steps,
            num_images_per_prompt=fastcomposer_args.num_images_per_prompt,
            #eta,
            #fastcomposer_args.generator,
            guidance_scale=fastcomposer_args.guidance_scale,
            controlnet_model = self.control_condition_model,
            controlnet_cond = controlnet_cond,
            controlnet_uncond = controlnet_un_cond,
            start_merge_step = fastcomposer_args.start_merge_step,
            prompt_embeds = encoder_hidden_states,
            prompt_embeds_text_only = encoder_hidden_states_text_only
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
    condition_image_path = "./data/dab_pose.jpg"
    condition_image = HWC3(np.array(Image.open(condition_image_path)))
    detected_map, _ = apply_openpose(resize_image(condition_image, 512))
    detected_map = HWC3(detected_map)
    image_resolution = 512
    condition_image = resize_image(condition_image, image_resolution)
    H, W, C = condition_image.shape
    
    detected_map = cv2.resize(detected_map, (512, 512), interpolation=cv2.INTER_NEAREST)

    num_samples = 1
    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    unique_token = "<|image|>"
    prompt = args.test_caption
    prompt_text_only = prompt.replace(unique_token, "")
    controlnet_cond = {"c_concat": [control], "c_crossattn": [sampler.get_learned_conditioning([prompt_text_only])]}
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    controlnet_un_cond = {"c_concat": [control], "c_crossattn": [sampler.get_learned_conditioning([n_prompt])]}

    shape = (1, 4, 512, 512)  # Your desired shape


    # Run combined sampling
    images = sampler.combined_sampling(
        cond=None,  # Will be set by FastComposer processing
        controlnet_cond=controlnet_cond,
        controlnet_un_cond=controlnet_un_cond,
        shape=shape,
        unconditional_guidance_scale=args.guidance_scale,
        fastcomposer_args=args
    )

    print(f"{images=}")

    # Save results
    for idx, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image = Image.fromarray(image.cpu().numpy())
        image.save(os.path.join(args.output_dir, f"combined_output_{idx}.png"))

if __name__ == "__main__":
    main()

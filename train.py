import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator

from data import PreppedIdentityPoseDataset
from knit import CombinedSampler

from fastcomposer.utils import parse_args
from fastcomposer.transforms import get_object_transforms
from fastcomposer.model import FastComposerModel

def recalibrate_control(sampler, train_loader, optimizer, num_epochs, device, unique_token = "<|image|>"):
    train_losses = []
    
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    uncond_cross_attn = [sampler.get_learned_conditioning([n_prompt])]

    for epoch in range(num_epochs):
        train_loss = 0.0

        for batch_idx, (prompt, fc_cond_args, ref_image, control_cond, control_uncond, ident_image) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            optimizer.zero_grad()

            prompt_text_only = prompt.replace(unique_token, "")
            control_cond['c_crossattn'] = [sampler.get_learned_conditioning([prompt_text_only])]
            control_uncond['c_crossattn'] = uncond_cross_attn
            shape = (1, 4, 512, 512)

            outputs = sampler.combined_sampling(
                cond=None,
                controlnet_cond=control_cond,
                controlnet_un_cond=control_uncond,
                shape=shape,
                unconditional_guidance_scale=8,
                fastcomposer_args = None,
                is_demo=False,
                input_ids=fc_cond_args['input_ids'].to(device)
                image_token_mask=fc_cond_args['image_token_mask'].to(device)
                all_object_pixel_values=fc_cond_args['object_pixel_values'].unsqueeze(0).to(device),
                num_objects = batch["num_objects"].unsqueeze(0).to(device)
            )

            #TODO: Get loss
            loss = None

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
            

if __name__ == "__main__":
    args = parse_args()
    weight_dtype="float32"
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cn_model_path = "./ControlNet/models/cldm_v15.yaml"
    cn_state_dict_path = "./ControlNet/models/control_sd15_openpose.pth"

    # Initialize combined sampler
    sampler = CombinedSapler(cn_model_path, cn_state_dict_path)
    sampler.setup_fastcomposer(args, accelerator, weight_dtype)
    

    train_loader = PreppedIdentityPoseDataset(
        "./prepped_idents/refs", 
        "./prepped_idents/poses", 
        "./prepped_idents/idents", 
        "./prepped_idents/prompts.txt",
        device,
        tokenizer=sampler.fc_tokenizer,

    )

    # TODO: Freeze parameters of all except for the sampler.control_condition_model

    epochs = 10
    optimizer = torch.optim.Adam(sampler.control_condition_model.parameters(), lr=0.0001)
    recalibrate_control(sampler, train_loader, optimizer, epochs, device)


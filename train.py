import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator

from data import PreppedIdentityPoseDataset
from knit import CombinedSampler

from facenet_pytorch import MTCNN, InceptionResnetV1
from evaluation.clip_eval import CLIPEvaluator
from evaluation.single_object.single_object_evaluation import compute_average_similarity 

from fastcomposer.utils import parse_args
from fastcomposer.transforms import get_object_transforms
from fastcomposer.model import FastComposerModel

def loss_fn(face_detector, face_similarity, text_evaluator, output, prompt, ref_image):
    #identity_similarity = compute_average_similarity(
    #    1, face_detector, face_similarity, output, ref_image
    #)

    prompt_similarity = text_evaluator.txt_to_img_similarity(
        prompt, output.unsqueeze(0) * 2.0 - 1.0
    )

    return prompt_similarity.sum()

def recalibrate_control(
    sampler, 
    train_loader, 
    optimizer, 
    num_epochs, 
    device,
    face_detector,
    face_similarity,
    text_evaluator,
    unique_token = "<|image|>"
):
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
                input_ids=fc_cond_args['input_ids'].to(device),
                image_token_mask=fc_cond_args['image_token_mask'].to(device),
                all_object_pixel_values=fc_cond_args['object_pixel_values'].unsqueeze(0).to(device),
                num_objects = fc_cond_args["num_objects"].unsqueeze(0).to(device),
                prompt=prompt,
                return_dict=False
            )

            print(f"{outputs.shape=}")

            # Get identity similarity
            loss = loss_fn(face_detector, face_similarity, text_evaluator, outputs, prompt, ref_image) 

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)


if __name__ == "__main__":
    args = parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    
    weight_dtype=torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cn_model_path = "./ControlNet/models/cldm_v15.yaml"
    cn_state_dict_path = "./ControlNet/models/control_sd15_openpose.pth"

    # Initialize combined sampler
    sampler = CombinedSampler(cn_model_path, cn_state_dict_path)
    sampler.setup_fastcomposer(args, accelerator, weight_dtype)
    

    train_loader = PreppedIdentityPoseDataset(
        "./prepped_idents/refs", 
        "./prepped_idents/poses", 
        "./prepped_idents/idents", 
        "./prepped_idents/prompts.txt",
        device,
        tokenizer=sampler.fc_tokenizer,
        object_transforms=sampler.object_transforms,
    )

    # TODO: Freeze parameters of all except for the sampler.control_condition_model

    face_detector = MTCNN(
        image_size = 160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=accelerator.device,
        keep_all=True,
    )
    face_similarity = (
        InceptionResnetV1(pretrained="vggface2").eval().to(accelerator.device)
    )

    text_evaluator = CLIPEvaluator(device=accelerator.device, clip_model="ViT-L/14")

    epochs = 10
    optimizer = torch.optim.Adam(sampler.control_condition_model.parameters(), lr=0.0001)
    recalibrate_control(
        sampler, train_loader, optimizer, epochs, device,
        face_detector, face_similarity, text_evaluator
    )


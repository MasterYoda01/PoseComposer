CAPTION="a man <|image|> is reading a book"
DEMO_NAME="lib_testing"

CUDA_VISIBLE_DEVICES=2 accelerate launch \
    --mixed_precision=fp16 \
    knit.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --finetuned_model_path model/fastcomposer \
    --test_reference_folder data/single_test \
    --test_caption "${CAPTION}" \
    --output_dir outputs/${DEMO_NAME} \
    --mixed_precision fp16 \
    --image_encoder_type clip \
    --image_encoder_name_or_path openai/clip-vit-large-patch14 \
    --num_image_tokens 1 \
    --max_num_objects 1 \
    --object_resolution 224 \
    --generate_height 512 \
    --generate_width 512 \
    --num_images_per_prompt 1 \
    --num_rows 1 \
    --seed 42 \
    --guidance_scale 5 \
    --inference_steps 100 \
    --start_merge_step 10 \
    --no_object_augmentation

#!/bin/bash

# Define variables for inputs
GROUND_TRUTH_FOLDER="poses"   # Folder containing ground truth pose PNG images
GENERATED_IMAGES_FOLDER="generated_images"  # Folder containing generated images
OUTPUT_FILE="oks_results.txt"  # File to save the OKS scores

# Run the OKS evaluation Python script
CUDA_VISIBLE_DEVICES=0 python oks_eval.py \
    --ground_truth_folder "${GROUND_TRUTH_FOLDER}" \
    --generated_images_folder "${GENERATED_IMAGES_FOLDER}" \
    --output_file "${OUTPUT_FILE}"

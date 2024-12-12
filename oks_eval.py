import os
import json
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pathlib import Path
from ControlNet.annotator.openpose import OpenposeDetector
from ControlNet.annotator.util import resize_image, HWC3

# Initialize OpenPose detector
apply_openpose = OpenposeDetector()

num_keypoints = 18  # Adjust if needed, based on your model
# Keypoint sigmas for 18 keypoints (example values, adjust as per your model's standard)
sigmas = np.array([
    0.26, 0.25, 0.25, 0.35, 0.35,
    0.79, 0.79, 0.72, 0.72,
    0.62, 0.62, 1.07, 1.07,
    0.87, 0.87, 0.89, 0.89,
    0.89
])

# Paths
ground_truth_folder = "poses"  # Folder with GT images and JSON files
generated_images_folder = "generated_images"  # Folder with predicted images


def calculate_oks(pred_keypoints, gt_keypoints, sigmas, bbox_area):
    """
    Calculate Object Keypoint Similarity (OKS).
    :param pred_keypoints: Predicted keypoints (np.array of shape [num_keypoints, 2])
    :param gt_keypoints: Ground truth keypoints (np.array of shape [num_keypoints, 2])
    :param sigmas: Standard deviations for keypoints
    :param bbox_area: Bounding box area of the object
    :return: OKS score
    """
    print(f"Ground truth keypoints shape: {gt_keypoints.shape}")
    print(f"Predicted keypoints shape: {pred_keypoints.shape}")
    assert pred_keypoints.shape == gt_keypoints.shape, "Keypoints shape mismatch"
    
    # Compute squared distances
    dists = np.sum((pred_keypoints - gt_keypoints) ** 2, axis=1)
    variances = (2 * (sigmas ** 2)) * (bbox_area + np.finfo(float).eps)
    oks_scores = np.exp(-dists / variances)
    return np.mean(oks_scores)

def extract_pose(image_path):
    """
    Extract predicted keypoints (x,y) from an image using OpenPose.
    Returns an np.array with shape (num_keypoints, 2).
    If no person is detected, returns zeros.
    """
    image = cv2.imread(image_path)
    image = resize_image(HWC3(image), 512)
    pose_image, pose_data = apply_openpose(image)

    candidate = pose_data.get('candidate', [])
    subset = pose_data.get('subset', [])

    candidate = np.array(candidate)
    subset = np.array(subset)

    if subset.size == 0:
        print(f"No person detected in image: {image_path}")
        return np.zeros((num_keypoints, 2))

    # Take the first person detected
    person = subset[0]
    keypoint_indexes = person[:num_keypoints].astype(int)

    keypoints = []
    for idx in keypoint_indexes:
        if idx < 0 or idx >= candidate.shape[0]:
            # Keypoint not found
            keypoints.append([0.0, 0.0])
        else:
            # candidate row might have more than 3 values, take first 3 for (x, y, score)
            x, y, score = candidate[idx][:3]
            keypoints.append([x, y])
    
    return np.array(keypoints)

def load_ground_truth_keypoints(json_path):
    """
    Load ground truth keypoints from a JSON file.
    JSON format (as provided):
    [
      {
        "people": [
          {
            "pose_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...]
          }
        ],
        "canvas_height": ...,
        "canvas_width": ...
      }
    ]

    Returns an np.array of shape (num_keypoints, 2).
    """
    if not os.path.exists(json_path):
        print(f"Ground truth JSON not found: {json_path}")
        return np.zeros((num_keypoints, 2))
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Assuming the first entry and first person:
    if not data or 'people' not in data[0] or len(data[0]['people']) == 0:
        print(f"No people data found in {json_path}")
        return np.zeros((num_keypoints, 2))
    
    kps = data[0]['people'][0]['pose_keypoints_2d']
    kps = np.array(kps).reshape(-1, 3)  # shape: (num_keypoints, 3)
    
    # Just extract x, y (ignore confidence)
    keypoints = kps[:, :2]

    # If the JSON does not match your expected number of keypoints, handle it here:
    if keypoints.shape[0] != num_keypoints:
        print(f"Number of keypoints in {json_path} ({keypoints.shape[0]}) does not match expected ({num_keypoints}).")
        # Pad or slice if needed
        # For now, assume it matches.
    
    return keypoints


# Automate OKS calculation
oks_scores = []
for gt_file in tqdm(sorted(os.listdir(ground_truth_folder))):
    if not gt_file.lower().endswith('.png'):
        continue  # skip non-image files

    # Corresponding JSON file for ground truth
    gt_json_file = os.path.join(ground_truth_folder, f"{Path(gt_file).stem}.json")
    gt_keypoints = load_ground_truth_keypoints(gt_json_file)

    # Predicted image
    image_file = os.path.join(generated_images_folder, f"{Path(gt_file).stem}.png")
    if not os.path.exists(image_file):
        print(f"Generated image not found for {gt_file}")
        continue

    pred_keypoints = extract_pose(image_file)

    if np.all(gt_keypoints == 0):
        print(f"No valid ground truth keypoints for {gt_file}, skipping OKS calculation.")
        continue

    # Compute bounding box area from ground truth keypoints
    bbox_area = (np.max(gt_keypoints[:, 0]) - np.min(gt_keypoints[:, 0])) * \
                (np.max(gt_keypoints[:, 1]) - np.min(gt_keypoints[:, 1]))

    # Calculate OKS
    oks_score = calculate_oks(pred_keypoints, gt_keypoints, sigmas, bbox_area)
    oks_scores.append(oks_score)

    print(f"{gt_file}: OKS = {oks_score:.4f}")

# Summary statistics
if oks_scores:
    print("\nEvaluation Summary:")
    print(f"Average OKS Score: {np.mean(oks_scores):.4f}")
    print(f"OKS Scores: {oks_scores}")
else:
    print("No OKS scores were computed.")

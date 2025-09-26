## Motivation

Current methods of adding conditions to text-to-image diffusion models require computationally expensive re-training or fine-tuning. We present PoseComposer, a method of adding multi-subject and spatially controlled image generation with forward passes only. The method uses pre-trained Primary and Secondary models. The Primary model drives image generation, and the Secondary model injects information into the U-Net of the Primary model. The result is a diffusion model that can generate images with personalized multi-subject image generation and controllable pose, without fine-tuning. Our method enables highly personalized content creation on consumer hardware.


### Environment Setup

Clone the repository with submodules.
```bash
git clone --recurse-submodules git@github.com:Cgreg2500/PoseComposer.git
```


#### Install FastComposer pre-reqs
```bash
python3 -m venv ./venv/
source ./venv/bin/activate
pip install torch torchvision torchaudio
pip install transformers==4.25.1 accelerate datasets evaluate diffusers==0.16.1 xformers triton scipy clip gradio facenet-pytorch
```

#### Install ControlNet pre-reqs (in venv)
```bash
pip install scikit-image huggingface_hub==0.25.0 gradio==3.16.2 omegaconf pytorch-lightning==1.5.0 einops opencv-python open_clip_torch matplotlib

python setup.py install
```

### Download the Pre-trained Models

```bash
mkdir -p model/fastcomposer ; cd model/fastcomposer
wget https://huggingface.co/mit-han-lab/fastcomposer/resolve/main/pytorch_model.bin
cd ../../ControlNet/annotator/ckpts
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth?download=true
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth?download=true
cd ../../models
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth
```

### Knitted FastComposer and ControlNet Inference

```bash
./scripts/run_knit.sh
```

### Generate Evaluation Images
```bash
./scripts/run_gen_evaluate.sh
```

### Run Image and Prompt evaluation
```bash
python ./single_object_evaluation.py
```

### Run OKS evaluation
```bash
./run_oks_eval.sg
```

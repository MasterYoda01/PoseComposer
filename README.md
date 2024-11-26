### Environment Setup

Clone the ControlNet repo first!

```bash
python3 -m venv ~/venv/fastcomposer
source ~/venv/fastcomposer/bin/activate
pip install torch torchvision torchaudio
pip install transformers==4.25.1 accelerate datasets evaluate diffusers==0.16.1 xformers triton scipy clip gradio facenet-pytorch

python setup.py install
```

### Download the Pre-trained Models

```bash
mkdir -p model/fastcomposer ; cd model/fastcomposer
wget https://huggingface.co/mit-han-lab/fastcomposer/resolve/main/pytorch_model.bin
```
Download the remaining models from huggingface for the ControlNet models

### Knitted FastComposer and ControlNet Inference

```bash
./scripts/run_knit.sh
```

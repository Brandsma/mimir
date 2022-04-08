import os

print("Installing required packages...")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.system("pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113")
os.system("pip install -r requirements.txt")
os.system("python -m spacy download en_core_web_sm")

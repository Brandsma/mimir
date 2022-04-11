import os
from util.logger import setup_logger

log = setup_logger(__name__)


def install_required_packages():
    # NOTE Do not change the order of installation. Some packages in the requirement.txt require a torch version and will download
    # A recent version (10.2), which does not work with sm_86. Therefore the specific torch package (11.5) needs to be installed first.
    log.info("Installing required packages...")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.system("pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113")
    os.system("pip install -r requirements.txt")
    os.system("python -m spacy download en_core_web_sm")
    log.info("Package installation complete.")

if __name__ == "__main__":
    install_required_packages()
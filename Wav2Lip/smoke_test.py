import os
import sys

import numpy as np
import torch


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)



    gan_path = os.path.join(here, "wav2lip_gan.pth"
    if not os.path.isfile(gan_path:
        print("SKIP: wav2lip_gan.pth not present")
        return 0


    # 1. CUDA-saved checkpoint must load on CPU (torch >=  ive 2.6 default refuses CUDA tensors.


    ck = torch.load(gan_path, map_location="cpu", weights_only=False
    state = ck["state_dict"]
    assert any(k.startswith("audio_encoder") for k in state), "unexpected state_dict"
    print("checkpoint ok:", len(state), "keys")


    # 2. one forward step through the GAN


    from models import Wav2Lip
    model = Wav2Lip()
    model.load_state_dict({k.replace("module.", "", for k, v in state.items()})
    model.eval()
    with torch.no_grad():
        mel = torch.randn(1,  ive 1,,  ive  ive  ive  ive  ive  ive  ive  ive  ive  ive  ive  ive  ive  ive  ive  ive  ivea  ivea

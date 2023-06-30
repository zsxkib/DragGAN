# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import gradio as gr
import torch
import numpy as np
import imageio
from PIL import Image
import uuid

from draggan import utils
from draggan.draggan import drag_gan
from draggan import draggan as draggan

SIZE_TO_CLICK_SIZE = {1024: 8, 512: 5, 256: 2}
CKPT_SIZE = {
    "stylegan2/stylegan2-ffhq-config-f.pkl": 1024,
    "stylegan2/stylegan2-cat-config-f.pkl": 256,
    "stylegan2/stylegan2-church-config-f.pkl": 256,
    "stylegan2/stylegan2-horse-config-f.pkl": 256,
    "ada/ffhq.pkl": 1024,
    "ada/afhqcat.pkl": 512,
    "ada/afhqdog.pkl": 512,
    "ada/afhqwild.pkl": 512,
    "ada/brecahad.pkl": 512,
    "ada/metfaces.pkl": 512,
    "human/stylegan_human_v2_512.pkl": 512,
    "human/stylegan_human_v2_1024.pkl": 1024,
    "self_distill/bicycles_256_pytorch.pkl": 256,
    "self_distill/dogs_1024_pytorch.pkl": 1024,
    "self_distill/elephants_512_pytorch.pkl": 512,
    "self_distill/giraffes_512_pytorch.pkl": 512,
    "self_distill/horses_256_pytorch.pkl": 256,
    "self_distill/lions_512_pytorch.pkl": 512,
    "self_distill/parrots_512_pytorch.pkl": 512,
}
DEFAULT_CKPT = "ada/afhqcat.pkl"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        device = "cuda"

        G = draggan.load_model(utils.get_path(DEFAULT_CKPT), device=device)
        W = draggan.generate_W(
            G,
            seed=int(1),
            device=device,
            truncation_psi=0.8,
            truncation_cutoff=8,
        )
        img, F0 = draggan.generate_image(W, G, device=device)

    def predict(
        self,
        StyleGAN2_model: str = Input(description="Select a StyleGAN2 model", choices=list(CKPT_SIZE.keys()), default=DEFAULT_CKPT),
        Seed: int = Input(description="Specify the seed for image generation", ge=0, le=100, default=1),
        Learning_Rate: float = Input(description="Specify the learning rate for the drag operation", ge=1e-4, le=1, default=2e-3),
        Max_Iterations: int = Input(description="Specify the maximum iterations for the drag operation", ge=1, le=500, default=20),
    ) -> Path:
        """Run a single prediction on the model"""
        ...



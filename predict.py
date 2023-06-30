# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import List
from cog import BasePredictor, Input, Path
import os
import shutil
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
OUTPUT_DIR = "./out/"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.device = "cuda"

        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.G = draggan.load_model(utils.get_path(DEFAULT_CKPT), device=self.device)
        self.W = draggan.generate_W(
            self.G,
            seed=int(1),
            device=self.device,
            truncation_psi=0.8,
            truncation_cutoff=8,
        )
        self.img, self.F0 = draggan.generate_image(self.W, self.G, device=self.device)
        self.model = {"G": self.G}

    def add_points_to_image(self, image, points, size=5):
        image = utils.draw_handle_target_points(image, points["handle"], points["target"], size)
        return image

    def on_drag(self, model, points, max_iters, state, size, mask, lr_box):
        if len(points["handle"]) == 0:
            raise Exception("You must select at least one handle point and target point.")
        if len(points["handle"]) != len(points["target"]):
            raise Exception(
                "You have uncompleted handle points, try to selct a target point or undo the handle point."
            )
        max_iters = int(max_iters)
        W = state["W"]

        handle_points = [torch.tensor(p, device=self.device).float() for p in points["handle"]]
        target_points = [torch.tensor(p, device=self.device).float() for p in points["target"]]

        if mask.get("mask", None) is not None:
            mask = Image.fromarray(mask["mask"]).convert("L")
            mask = np.array(mask) == 255

            mask = torch.from_numpy(mask).float().to(self.device)
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = None

        step = 0
        for image, W, handle_points in drag_gan(
            W, model["G"], handle_points, target_points, mask, max_iters=max_iters, lr=lr_box
        ):
            points["handle"] = [p.cpu().numpy().astype("int") for p in handle_points]
            image = self.add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])

            state["history"].append(image)
            step += 1

            output_path = os.path.join(OUTPUT_DIR, f"output_{step}.png")
            Image.fromarray(image).save(output_path)

            yield (image, state, step)

    def predict(
        self,
        # stylegan2_model: str = Input(
        #     description="Select a StyleGAN2 model", choices=list(CKPT_SIZE.keys()), default=DEFAULT_CKPT
        # ),
        # seed: int = Input(description="Specify the seed for image generation", ge=0, le=100, default=1),
        lr_box: float = Input(
            description="Specify the learning rate for the drag operation", ge=1e-4, le=1, default=2e-3
        ),
        max_iters: int = Input(
            description="Specify the maximum iterations for the drag operation", ge=1, le=500, default=20
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        points = {"target": [[309, 57]], "handle": [[379, 247]]}  # TODO: add as param
        state = {"W": self.W, "history": []}
        size = 512  # TODO: add as param
        mask = {}

        frames = []
        for image, _, _ in self.on_drag(self.model, points, max_iters, state, size, mask, lr_box):
            frames.append(image)

        # use output_frames to create video
        video_name = f"{OUTPUT_DIR}video_{uuid.uuid4()}.mp4"
        imageio.mimsave(video_name, frames)

        return Path(video_name)

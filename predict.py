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
from tqdm import tqdm

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
DEFAULT_CKPT = "ada/afhqcat.pkl"  # NOTE: Not wrapped in Path(.) on purpose
OUTPUT_DIR = Path("./out")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.init_output_dir()
        self.load_model(DEFAULT_CKPT)

    def init_output_dir(self):
        for child in OUTPUT_DIR.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def load_model(self, checkpoint=DEFAULT_CKPT):
        self.G = draggan.load_model(utils.get_path(checkpoint), device=self.device)
        self.W = draggan.generate_W(
            self.G,
            seed=int(1),
            device=self.device,
            truncation_psi=0.8,
            truncation_cutoff=8,
        )
        self.img, self.F0 = draggan.generate_image(self.W, self.G, device=self.device)
        self.model = {"G": self.G}
        self.current_checkpoint = checkpoint

    def add_points_to_image(self, image, points, point_size=5):
        image = utils.draw_handle_target_points(image, points["handle"], points["target"], point_size)
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
            image = self.add_points_to_image(image, points, point_size=SIZE_TO_CLICK_SIZE[size])

            state["history"].append(image)
            step += 1

            output_path = OUTPUT_DIR / f"output_{step}.png"
            Image.fromarray(image).save(output_path)

            yield (image, state, step)

    def predict(
        self,
        just_render_first_frame: bool = Input(
            description="If true, only the first frame will be rendered, providing a preview of the initial and final positions.",
            default=False,
        ),
        stylegan2_model: str = Input(
            description="The chosen StyleGAN2 model to perform the operation.",
            choices=list(CKPT_SIZE.keys()),
            default=DEFAULT_CKPT,
        ),
        handle_y_pct: float = Input(
            description="Set the y-coordinate percentage for the starting position. Lower values represent higher positions on the screen.",
            ge=0,
            le=100,
            default=50,
        ),
        handle_x_pct: float = Input(
            description="Set the x-coordinate percentage for the starting position. Lower values represent more leftward positions on the screen.",
            ge=0,
            le=100,
            default=50,
        ),
        target_y_pct: float = Input(
            description="Set the y-coordinate percentage for the final position. Lower values represent higher positions on the screen.",
            ge=0,
            le=100,
            default=40,
        ),
        target_x_pct: float = Input(
            description="Set the x-coordinate percentage for the final position. Lower values represent more leftward positions on the screen.",
            ge=0,
            le=100,
            default=10,
        ),
        lr_box: float = Input(
            description="Set the learning rate for the operation, which controls how quickly the model learns to drag the path from the initial to the final position.",
            ge=1e-4,
            le=1,
            default=3e-3,
        ),
        max_iters: int = Input(
            description="The maximum number of iterations allowed for the operation, limiting how long the path dragging can continue.",
            ge=1,
            le=100,
            default=20,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if self.current_checkpoint != stylegan2_model:
            self.load_model(stylegan2_model)
        self.init_output_dir()

        size = CKPT_SIZE[stylegan2_model]
        handle_y = int(size * handle_y_pct / 100)
        handle_x = int(size * handle_x_pct / 100)
        target_y = int(size * target_y_pct / 100)
        target_x = int(size * target_x_pct / 100)

        points = {"target": [[target_y, target_x]], "handle": [[handle_y, handle_x]]}

        state = {"W": self.W, "history": []}
        mask = {}
        max_iters = 1 if just_render_first_frame else max_iters
        lr_box = 0 if just_render_first_frame else lr_box
        f = (
            OUTPUT_DIR / f"output_1.png"
            if just_render_first_frame
            else OUTPUT_DIR / f"video_{uuid.uuid4()}.mp4"
        )

        frames = [
            image
            for image, _, _ in tqdm(
                self.on_drag(self.model, points, max_iters, state, size, mask, lr_box),
                total=max_iters,
            )
        ]
        if not just_render_first_frame:
            imageio.mimsave(f, frames)

        return Path(f)

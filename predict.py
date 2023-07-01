# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

# Standard library imports
import shutil
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

# Related third-party imports
import imageio
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage  # For Typing PIL.Image.Image
from cog import BasePredictor, Input, Path
from tqdm import tqdm

# Local application/library specific imports
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
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.device = "cuda"
        self.output_dir = OUTPUT_DIR
        self.init_output_dir()
        self.load_model(DEFAULT_CKPT)
        self.show_points_and_arrows = True

    def init_output_dir(self) -> None:
        """Creates/Recreates the output folder where all the images and videos go"""

        for child in self.output_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, checkpoint: str = DEFAULT_CKPT) -> None:
        """Loads the model given a checkpoint path"""

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

    def add_points_to_image(
        self,
        image: PILImage,
        points: Dict[str, List[List[int]]],
        point_size: int = 5,
    ) -> np.ndarray:
        """Adds the points, lines, and arrows to the image"""

        return (
            utils.draw_handle_target_points(image, points["handle"], points["target"], point_size)
            if self.show_points_and_arrows
            else np.array(image)
        )

    def on_drag(
        self,
        points: Dict[str, List[List[int]]],
        max_iters: int,
        state: Dict[str, Any],
        size: int,
        mask: Dict[str, Optional[Any]],
        lr_box: float,
    ) -> Generator[Tuple[PILImage, Dict[str, Any], int], Any, None]:
        """The magic drag function!"""

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
            W,
            self.model["G"],
            handle_points,
            target_points,
            mask,
            max_iters=max_iters,
            lr=lr_box,
        ):
            points["handle"] = [p.cpu().numpy().astype("int") for p in handle_points]
            image = self.add_points_to_image(image, points, point_size=SIZE_TO_CLICK_SIZE[size])

            state["history"].append(image)
            step += 1

            output_path = self.output_dir / f"output_{step}.png"
            Image.fromarray(image).save(output_path)

            yield (image, state, step)

    def predict(
        self,
        only_render_first_frame: bool = Input(
            description="If true, only the first frame will be rendered, providing a preview of the initial and final positions. Remember to check `show_points_and_arrows`!",
            default=False,
        ),
        stylegan2_model: str = Input(
            description="The chosen StyleGAN2 model to perform the operation.",
            choices=list(CKPT_SIZE.keys()),
            default=DEFAULT_CKPT,
        ),
        source_x_percentage: float = Input(
            description="Percentage defining the starting x-coordinate from the left. Higher values mean further to the right on the screen.",
            ge=0,
            le=100,
            default=50,
        ),
        source_y_percentage: float = Input(
            description="Percentage defining the starting y-coordinate from the bottom. Higher values mean higher up on the screen.",
            ge=0,
            le=100,
            default=50,
        ),
        target_x_percentage: float = Input(
            description="Percentage defining the final x-coordinate from the left. Higher values mean further to the right on the screen.",
            ge=0,
            le=100,
            default=10,
        ),
        target_y_percentage: float = Input(
            description="Percentage defining the final y-coordinate from the bottom. Higher values mean higher up on the screen.",
            ge=0,
            le=100,
            default=40,
        ),
        learning_rate: float = Input(
            description="Set the learning rate for the operation, which controls how quickly the model learns to drag the path from the initial to the final position.",
            ge=1e-4,
            le=1,
            default=3e-3,
        ),
        maximum_n_iterations: int = Input(
            description="The maximum number of iterations allowed for the operation, limiting how long the path dragging can continue.",
            ge=1,
            le=100,
            default=20,
        ),
        show_points_and_arrows: bool = Input(
            description="Toggles the display of arrows and points denoting the interpolation path in the generated video.",
            default=True,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if self.current_checkpoint != stylegan2_model:
            self.load_model(stylegan2_model)
        self.init_output_dir()
        self.show_points_and_arrows = show_points_and_arrows

        size = CKPT_SIZE[stylegan2_model]
        # Origin is bottom left
        handle_y, target_y = (
            int(size * (100 - source_y_percentage) / 100),
            int(size * (100 - target_y_percentage) / 100),
        )
        handle_x, target_x = (
            int(size * source_x_percentage / 100),
            int(size * target_x_percentage / 100),
        )
        points = {"target": [[target_y, target_x]], "handle": [[handle_y, handle_x]]}
        state = {"W": self.W, "history": []}
        mask = {}
        max_iters = 1 if only_render_first_frame else maximum_n_iterations
        lr_box = 0 if only_render_first_frame else learning_rate
        f = (
            self.output_dir / f"output_1.png"
            if only_render_first_frame
            else self.output_dir / f"video_{uuid.uuid4()}.mp4"
        )

        frames = [
            image
            for image, _, _ in tqdm(
                self.on_drag(points, max_iters, state, size, mask, lr_box),
                total=max_iters,
            )
        ]
        if not only_render_first_frame:
            imageio.mimsave(f, frames)

        return Path(f)

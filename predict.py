# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

# Standard library imports
import re
import sys
from random import randint
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
    # "human/stylegan_human_v2_512.pkl": 512, # TODO: Doesn't work, Fix `utils.create_square_mask`
    # "human/stylegan_human_v2_1024.pkl": 1024, # TODO: Doesn't work, Fix `utils.create_square_mask`
    "self_distill/bicycles_256_pytorch.pkl": 256,
    "self_distill/dogs_1024_pytorch.pkl": 1024,
    "self_distill/elephants_512_pytorch.pkl": 512,
    "self_distill/giraffes_512_pytorch.pkl": 512,
    "self_distill/horses_256_pytorch.pkl": 256,
    "self_distill/lions_512_pytorch.pkl": 512,
    "self_distill/parrots_512_pytorch.pkl": 512,
}
DEFAULT_CKPT = "ada/afhqcat.pkl"  # NOTE: Not wrapped in Path(.) on purpose
OUTPUT_DIR = Path("./out/")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.seed = int(1)
        self.device = "cuda"
        self.output_dir = OUTPUT_DIR
        self.init_output_dir()
        self.load_model(DEFAULT_CKPT)
        self.show_points_and_arrows = True
        # TODO: Change to lazy loading (cache type thing)
        # for checkpoint in CKPT_SIZE.keys():
        #     self.load_model(checkpoint)

    def init_output_dir(self) -> None:
        """Creates/Recreates the output folder where all the images and videos go"""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for child in self.output_dir.iterdir():
            if child.is_file():
                child.unlink()

    def load_model(self, checkpoint: str = DEFAULT_CKPT) -> None:
        """Loads the model given a checkpoint path"""

        self.G = draggan.load_model(utils.get_path(checkpoint), device=self.device)
        self.W = draggan.generate_W(
            self.G,
            seed=self.seed,
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
            utils.draw_handle_target_points(
                image, points["handle"], points["target"], point_size
            )
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
            raise Exception(
                "You must select at least one handle point and target point."
            )
        if len(points["handle"]) != len(points["target"]):
            raise Exception(
                "You have uncompleted handle points, try to selct a target point or undo the handle point."
            )
        max_iters = int(max_iters)
        W = state["W"]

        handle_points = [
            torch.tensor(p, device=self.device).float() for p in points["handle"]
        ]
        target_points = [
            torch.tensor(p, device=self.device).float() for p in points["target"]
        ]

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
            image = self.add_points_to_image(
                image, points, point_size=SIZE_TO_CLICK_SIZE[size]
            )

            state["history"].append(image)
            step += 1

            output_path = self.output_dir / f"output_{step}.png"
            Image.fromarray(image).save(output_path)

            yield (image, state, step)

    def parse_input(self, input_str: str) -> List[List[float]]:
        """
        Parses a string of comma separated numbers possibly surrounded by various types of brackets.
        Converts them into list of pairs (as list of floats).

        Parameters
        ----------
        input_str : str
            The string to parse. It should contain comma separated numbers, possibly surrounded by various types of brackets.

        Returns
        -------
        List[List[float]]
            A list of lists, where each sub-list contains a pair of floats.

        Notes
        -----
        The function ignores invalid characters (non-numbers and non-commas) and handles numbers without a leading digit.

        Examples
        --------
        Normal cases:
        >>> parse_input("[(1,2), (3,4)]")
        [[1.0, 2.0], [3.0, 4.0]]

        >>> parse_input("[[1,2], [3,4]]")
        [[1.0, 2.0], [3.0, 4.0]]

        Multiple nested brackets:
        >>> parse_input("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]")
        [[1, 2], [3, 4], [5, 6], [7, 8]]

        Negative numbers:
        >>> parse_input("[-1, -2]")
        [[-1, -2]]

        Decimal points without leading digits:
        >>> parse_input("[.5, 1]")
        [[0.5, 1]]

        Whitespace variations:
        >>> parse_input("[1,     2]")
        [[1, 2]]

        >>> parse_input("[(1,  2),  (3,4)]")
        [[1, 2], [3, 4]]

        Invalid characters:
        >>> parse_input("a,b")
        [[0.0, 0.0]]
        """
        # sanitize the input, removing all brackets and only leaving numbers, commas, and minus signs
        sanitized = re.sub(r"[^\d.,-]", "", input_str)

        # split the sanitized input string by commas
        numbers_str = sanitized.split(",")

        # convert strings to int or float as necessary
        # Exclude empty strings and convert to float even for whole numbers for consistency
        numbers = [float(x) if x else 0.0 for x in numbers_str]

        # group them into pairs and convert them into the format that you need (list of lists)
        # This ensures we always have pairs of numbers even if an odd number of numbers is provided
        return [
            numbers[i : i + 2]
            for i in range(0, len(numbers), 2)
            if len(numbers[i : i + 2]) == 2
        ]

    def predict(
        self,
        only_render_first_frame: bool = Input(
            description="If true, only the first frame will be rendered, providing a preview of the initial and final positions. Remember to check `show_points_and_arrows`!",
            default=False,
        ),
        show_points_and_arrows: bool = Input(
            description="Toggles the display of arrows and points denoting the interpolation path in the generated video.",
            default=True,
        ),
        stylegan2_model: str = Input(
            description="The chosen StyleGAN2 model to perform the operation.",
            choices=list(CKPT_SIZE.keys()),
            default=DEFAULT_CKPT,
        ),
        source_pixel_coords: str = Input(
            description="Pixel values defining the starting coordinates. String should be formatted as '(x, y)', where x is the pixel count from the left, and y is the pixel count from the bottom. Higher x values mean further to the right on the screen, higher y values mean higher up on the screen.",
            default="[(200, 100)]",
        ),
        target_pixel_coords: str = Input(
            description="Pixel values defining the final coordinates. String should be formatted as '(x, y)', where x is the pixel count from the left, and y is the pixel count from the bottom. Higher x values mean further to the right on the screen, higher y values mean higher up on the screen.",
            default="[(100, 200)]",
        ),
        learning_rate: float = Input(
            description="Set the learning rate for the operation, which controls how quickly the model learns to drag the path from the initial to the final position.",
            ge=1e-4,
            le=0.15,
            default=3e-3,
        ),
        maximum_n_iterations: int = Input(
            description="The maximum number of iterations allowed for the operation, limiting how long the path dragging can continue.",
            ge=1,
            le=100,
            default=20,
        ),
        seed: int = Input(
            description="Changes init image via the random seed. Set to -1 for a random seed, otherwise provide an integer value.",
            default=-1,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        self.seed = randint(0, sys.maxsize) if seed == -1 else int(seed)

        if self.current_checkpoint != stylegan2_model:
            self.load_model(stylegan2_model)
        self.init_output_dir()
        self.show_points_and_arrows = show_points_and_arrows

        size = CKPT_SIZE[stylegan2_model]
        points = {
            "target": self.parse_input(target_pixel_coords),
            "handle": self.parse_input(source_pixel_coords),
        }
        state = {"W": self.W, "history": []}
        mask = {}
        max_iters = 1 if only_render_first_frame else maximum_n_iterations
        lr_box = 0 if only_render_first_frame else learning_rate
        f = (
            self.output_dir / f"output_1.png"
            if only_render_first_frame
            else self.output_dir / f"video.mp4"
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

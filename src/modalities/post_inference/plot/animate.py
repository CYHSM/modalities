from pathlib import Path

from PIL import Image


def create_gif(output_dir="output", fps=2):
    output_path = Path(output_dir)
    image_files = sorted(output_path.glob("step_*.png"))

    if not image_files:
        raise ValueError(f"No step_*.png files found in {output_path}")

    images = [Image.open(f) for f in image_files]

    duration = int(1000 / fps)
    gif_path = output_path / "animation.gif"

    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

    print(f"Created: {gif_path} ({len(images)} frames, {duration}ms per frame)")

    return gif_path

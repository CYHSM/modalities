from PIL import Image
from pathlib import Path
import numpy as np

def create_gif(image_dir, output_path, duration=500, loop=0):
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("step_*.png"))
    
    if not image_files:
        raise ValueError(f"No step_*.png files found in {image_dir}")
    
    images = [Image.open(f) for f in image_files]
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )
    
    print(f"Created GIF: {output_path} ({len(images)} frames)")

def create_video(image_dir, output_path, fps=2):
    import cv2
    
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("step_*.png"))
    
    if not image_files:
        raise ValueError(f"No step_*.png files found in {image_dir}")
    
    first_img = cv2.imread(str(image_files[0]))
    height, width, layers = first_img.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for image_file in image_files:
        img = cv2.imread(str(image_file))
        video.write(img)
    
    video.release()
    cv2.destroyAllWindows()
    
    print(f"Created video: {output_path} ({len(image_files)} frames)")

def create_comparison_grid(dirs, output_path, duration=500):
    dirs = [Path(d) for d in dirs]
    
    all_steps = []
    for step_idx in range(100):
        step_images = []
        for dir_path in dirs:
            step_file = dir_path / f"step_{step_idx:03d}.png"
            if not step_file.exists():
                break
            step_images.append(Image.open(step_file))
        
        if len(step_images) != len(dirs):
            break
        
        widths = [img.width for img in step_images]
        heights = [img.height for img in step_images]
        
        combined_width = sum(widths)
        max_height = max(heights)
        
        combined = Image.new('RGB', (combined_width, max_height))
        
        x_offset = 0
        for img in step_images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
        
        all_steps.append(combined)
    
    if all_steps:
        all_steps[0].save(
            output_path,
            save_all=True,
            append_images=all_steps[1:],
            duration=duration,
            loop=0
        )
        print(f"Created comparison GIF: {output_path} ({len(all_steps)} frames)")

if __name__ == "__main__":
    #create_gif("viz_gray", "activation_gray.gif", duration=300)
    #create_gif("viz_viridis", "activation_viridis.gif", duration=300)
    # create_gif("viz_inferno", "activation_inferno.gif", duration=50)

    normalization = "nonorm"
    colormap = "inferno"
    output_dir = f"/raid/s3/opengptx/mfrey/cp_analysis/inference_viz/viz_{colormap}_{normalization}"
    gif_path = f"/raid/s3/opengptx/mfrey/cp_analysis/inference_viz/activation_{colormap}_{normalization}_50ms.gif"
    create_gif(output_dir, gif_path, duration=50)
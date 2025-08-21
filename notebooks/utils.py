import os
import glob
import tqdm
import matplotlib.pyplot as plt
from PIL import Image as Img
from IPython.display import Image, display

def generate_output_gif(graph, data, base_path, gif_name="classification.gif", duration=50, cleanup=True, show=True, title="Cuvis AI Result"):
    """
    Generates and saves a GIF of classification results over a dataset.

    Args:
        graph: A model with a `.forward()` method that returns an image-like output.
        data: A list or array-like object of inputs to the model.
        base_path: Directory to save frames and the resulting GIF.
        gif_name: Name of the output GIF file.
        duration: Duration between frames in milliseconds.
        cleanup: If True, deletes intermediate PNG files after creating the GIF.
        show: If True, displays the resulting GIF in a Jupyter notebook.
        title: Title to annotate the results window with
    Returns:
        Path to the generated GIF.
    """
    os.makedirs(base_path, exist_ok=True)

    # Generate frames
    for i in tqdm.tqdm(range(len(data)), desc="Generating frames"):
        res_show = graph.forward(*data[i:i+1])
        plt.figure()
        plt.imshow(res_show[0, :, :, :])
        plt.title(title)
        plt.axis('off')
        frame_path = os.path.join(base_path, f"frame_{i}.png")
        plt.savefig(frame_path)
        plt.close()

    # Create GIF
    frames = []
    for i in range(len(data)):
        frame_path = os.path.join(base_path, f"frame_{i}.png")
        frames.append(Img.open(frame_path))
        
    gif_path = os.path.join(base_path, gif_name)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

    # Optionally clean up intermediate frames
    if cleanup:
        for f in glob.glob(os.path.join(base_path, "frame_*.png")):
            os.remove(f)

    # Optionally show the GIF
    if show:
        display(Image(filename=gif_path))

    return gif_path

import pathlib
import math
import io

from skimage.io import imread
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import display
from ipywidgets import GridspecLayout
import ipywidgets as widgets
from PIL import Image



def change_colormap(image_path: pathlib.Path, cmap='RdYlBu'):
    feature = imread(image_path)
    cm = plt.get_cmap(cmap)
    colored_image = cm(-feature)
    colored_feature = Image.fromarray(np.uint8(colored_image * 255))
    imgByteArr = io.BytesIO()
    colored_feature.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def display_image_and_references(image_path):
    image_folder = image_path.parent
    print(image_folder)

    other_images = [
        f for f in image_folder.glob("img_ndwi*.png")
        if f.is_file() and f != image_path
    ]
    other_images.extend(image_folder.glob("bg_ndwi*.png"))

    n_col = 4
    n_row = max(math.ceil(len(other_images) / n_col), 1)
    grid = GridspecLayout(n_row, n_col)

    for i in range(n_row):
        for j in range(n_col):
            img_index = i * n_col + j
            if img_index >= len(other_images): break
            image = other_images[img_index]
            grid[i, j] = widgets.VBox([
                widgets.Label(f"Image {image.name}"),
                widgets.Image(value=change_colormap(image),
                              layout=widgets.Layout(width='200px', height='200px')),
            ])

    image_display = widgets.VBox([
        widgets.VBox([
            widgets.Label(f"folder: {image_folder}"),
            widgets.Label(f"all other images of the same loc"),
            grid,
            widgets.Label(f"image to label: {image_path.name}"),
            widgets.Image(value=change_colormap(image_path), object_fit='none',
                          layout=widgets.Layout(width='300px', height='300px'))
        ]),

    ])
    display(image_display)
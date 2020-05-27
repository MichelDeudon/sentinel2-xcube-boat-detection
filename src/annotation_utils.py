import pathlib
import math
import io
from typing import Dict, List, Tuple
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import display
from ipywidgets import GridspecLayout
import ipywidgets as widgets
from PIL import Image




def transform_img_path(image_path: pathlib.Path, cmap='RdYlBu', reverse=True):
    """
    read image from file, map it to the given cmap and return the byte array of it
    :param image_path: path of the image
    :param cmap: color map, followed matplotlib format
    :param reverse: if True will flip the color map and display the false color
    :return: transformed image as byte array
    """
    feature = imread(image_path)
    cm = plt.get_cmap(cmap)
    if reverse:
        colored_image = cm(-feature)
    else:
        colored_image = cm(feature)
    colored_feature = Image.fromarray(np.uint8(colored_image))
    imgByteArr = io.BytesIO()
    colored_feature.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


# def transform_img_nparray(image: np.ndarray, cmap='gray'):

#     """
#     read image from file, map it to the given cmap and return the byte array of it
#     :param image: path of the image
#     :param cmap: color map, followed matplotlib format
#     :return: transformed image as byte array
#     """
#     cm = plt.get_cmap(cmap)
#     colored_image = cm(image)
#     scaled_image = (colored_image - np.min(colored_image)) / (np.max(colored_image) - np.min(colored_image))
#     colored_feature = Image.fromarray(np.uint8(scaled_image * 255))
#     imgByteArr = io.BytesIO()
#     colored_feature.save(imgByteArr, format='PNG')
#     imgByteArr = imgByteArr.getvalue()
#     return imgByteArr



def display_image_and_references(image_path: Path):
    """
    display function of superintendent. display img_ndwi and all the other img_ndwi of the same location as reference to labeler
    :param image_path: image path
    :return: ipython display handle
    """
    image_folder = image_path.parent

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
                widgets.Label("Image {0}".format(image.name)),
                widgets.Image(value=transform_img_path(image),
                              layout=widgets.Layout(width='200px', height='200px')),
            ])

    image_display = widgets.VBox([
        widgets.VBox([
            widgets.Label("folder: {0}".format(image_folder)),
            widgets.Label("all other images of the same loc"),
            grid,
            widgets.Label("image to label: {0}".format(image_path.name)),
            widgets.Image(value=transform_img_path(image_path), object_fit='none',
                          layout=widgets.Layout(width='300px', height='300px'))
        ]),

    ])
    display(image_display)

def display_heatmap_prediction(image_titles: List[Tuple[np.ndarray, str]]):
    """
    display y_true, heatmap, y_hat of a model
    :param image_titles: a list of tuple (image, title). Image as ndarray and title of the image
    :return: ipython display handle
    """
#     fig, axs = plt.subplots(1,3, figsize=(10,5))
    fig = plt.figure(figsize=(10,5))   

    plt.subplot(1,3,1)
    plt.imshow(image_titles[0][0], cmap='gray')
    plt.title(image_titles[0][1])
    plt.xticks([])
    plt.yticks([])
                
    plt.subplot(1,3,2)
               
    plt.imshow(image_titles[1][0], cmap='gray')
    plt.title(image_titles[1][1])
    plt.xticks([])
    plt.yticks([])
                
    plt.subplot(1,3,3)
    plt.imshow(image_titles[2][0], cmap='gray')
    plt.title(image_titles[2][1])
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    display(fig)
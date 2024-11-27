from io import BytesIO

import cv2
from numpy import load as np_load, save as np_save, zeros as np_zeros, uint8 as np_uint8


segmentation_index_to_color_map = { # BGR
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 0, 0),
}


def save_np_arr_to_buffer(np_arr):
    data_buffer = BytesIO()
    
    np_save(data_buffer, np_arr)
    return data_buffer


def load_np_arr_from_stream(stream_content):
    buffer = BytesIO(stream_content)
    
    np_arr = np_load(buffer)
    return np_arr


def multi_label_segmentation_to_colored_image(segmentation_arr):
    height, width = segmentation_arr.shape
    rgb_image = np_zeros((height, width, 3), dtype=np_uint8)
    
    for key, color in segmentation_index_to_color_map.items():
        rgb_image[segmentation_arr == key] = color

    return rgb_image


def save_multi_label_segmentation_as_image(output_path, segmentation_arr):
    rgb_image = multi_label_segmentation_to_colored_image(segmentation_arr)
    cv2.imwrite(output_path, rgb_image)

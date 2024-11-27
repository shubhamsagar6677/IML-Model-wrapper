from json import loads as json_loads
from os import makedirs
from os.path import (
    dirname as path_dirname,
    join as path_join,
)
from requests import Session
from random import randint

import cv2
import unittest
import numpy as np

from utils import (
    load_np_arr_from_stream,
    save_multi_label_segmentation_as_image,
    save_np_arr_to_buffer,
)


unittest.TestLoader.sortTestMethodsUsing = None

requests_session = Session()
api_base_url = "http://lipikar.cse.iitd.ac.in:7901/api/interactive-model-training"

inputs_dir = path_join(path_dirname(__file__), "inputs")
outputs_dir = path_join(path_dirname(__file__), "outputs")
makedirs(outputs_dir, exist_ok=True)


class TestSegment(unittest.TestCase):
    def setUp(self):
        self.endpoint = f"{api_base_url}/segment"
    
    def test_invalid_numpy_array(self):
        response = requests_session.post(self.endpoint, data=b'this_sure_isnt_a_numpy_array')
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.headers.get('Content-Type', None), 'application/json')

        response_data = response.json()
        self.assertEqual(response_data.get('success'), False)
        self.assertEqual(response_data.get('error', {}).get('code'), 'request_body_is_not_a_valid_numpy_array')
    
    def run_unsuccessfully_on_np_arrs(self, input_np_arr, expected_error_code):
        request_data_buffer = save_np_arr_to_buffer(input_np_arr)

        response = requests_session.post(self.endpoint, data=request_data_buffer.getvalue())
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.headers.get('Content-Type', None), 'application/json')

        response_data = response.json()
        self.assertEqual(response_data.get('success'), False)
        self.assertEqual(response_data.get('error', {}).get('code'), expected_error_code)

    def test_non_4d_numpy_array(self):
        all_inputs_np_arr = np.random.rand(5, 6)
        self.run_unsuccessfully_on_np_arrs(all_inputs_np_arr, "request_body_numpy_array_is_not_4d")
    
    def test_non_uint8_numpy_array(self):
        # Test uint16
        all_inputs_np_arr_1 = (np.random.rand(5, 6, 8, 4) * 65535).astype(np.uint16)
        self.run_unsuccessfully_on_np_arrs(all_inputs_np_arr_1, "request_body_numpy_array_is_not_of_type_uint8")

        # Test float
        all_inputs_np_arr_2 = np.random.rand(5, 6, 8, 4).astype(np.float64)
        self.run_unsuccessfully_on_np_arrs(all_inputs_np_arr_2, "request_body_numpy_array_is_not_of_type_uint8")

        # Test bool
        all_inputs_np_arr_3 = (np.random.rand(5, 6, 8, 4) > 0.5).astype(bool)
        self.run_unsuccessfully_on_np_arrs(all_inputs_np_arr_3, "request_body_numpy_array_is_not_of_type_uint8")
    
    def test_non_3_channel_images_numpy_array(self):
        all_inputs_np_arr = np.random.rand(5, 6, 7, 5).astype(np.uint8)
        self.run_unsuccessfully_on_np_arrs(
            all_inputs_np_arr,
            "request_body_numpy_array_is_not_stack_of_3_channel_images"
        )
    
    def run_successfully_on_np_arrs(self, image_np_arrs, draw_outputs=False, output_filename_prefix=""):
        num_inputs = len(image_np_arrs)
        image_height, image_width, _image_cs = image_np_arrs[0].shape

        inputs_np_arr = np.stack(image_np_arrs, axis=0)
        
        request_data_buffer = save_np_arr_to_buffer(inputs_np_arr)

        response = requests_session.post(self.endpoint, data=request_data_buffer.getvalue())
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get('Content-Type', None), 'application/octet-stream')

        masks_arr = load_np_arr_from_stream(response.content)

        return masks_arr



    def test_4d_np_arr_single_input(self):
        num_inputs = 1
        image_height, image_width = randint(100, 1280), randint(100, 720)

        all_images = []
        for _i in range(num_inputs):
            random_image_arr = np.random.randint(0, 255, size=(image_height, image_width, 3), dtype=np.uint8)
            all_images.append(random_image_arr)

        self.run_successfully_on_np_arrs(all_images)
    
    def test_4d_np_arr_multiple_inputs(self):
        num_inputs = randint(2, 10)
        image_height, image_width = randint(100, 1280), randint(100, 720)

        all_images = []
        for _i in range(num_inputs):
            random_image_arr = np.random.randint(0, 255, size=(image_height, image_width, 3), dtype=np.uint8)
            all_images.append(random_image_arr)

        self.run_successfully_on_np_arrs(all_images)

    def test_single_image(self):
        image_arr = cv2.imread(path_join(inputs_dir, "1.jpg"))
        all_images = [image_arr]
        
        return self.run_successfully_on_np_arrs(all_images, True, "segment_test_single-")
    
    def test_multiple_images(self):
        image_1_arr = cv2.imread(path_join(inputs_dir, "1.jpg"))
        image_2_arr = cv2.imread(path_join(inputs_dir, "2.jpg"))
        all_images = [image_1_arr, image_2_arr]
        
        self.run_successfully_on_np_arrs(all_images, True, "segment_test_multiple-")


class TestInteractiveSegment(unittest.TestCase):
    def setUp(self):
        self.endpoint = f"{api_base_url}/interactive-segment"
    
    def test_invalid_numpy_array(self):
        response = requests_session.post(self.endpoint, data=b'this_sure_isnt_a_numpy_array')
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.headers.get('Content-Type', None), 'application/json')

        response_data = response.json()
        self.assertEqual(response_data.get('success'), False)
        self.assertEqual(response_data.get('error', {}).get('code'), 'request_body_is_not_a_valid_numpy_array')

    def run_unsuccessfully_on_np_arrs(self, input_np_arr, expected_error_code):
        request_data_buffer = save_np_arr_to_buffer(input_np_arr)

        response = requests_session.post(self.endpoint, data=request_data_buffer.getvalue())
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.headers.get('Content-Type', None), 'application/json')

        response_data = response.json()
        self.assertEqual(response_data.get('success'), False)
        self.assertEqual(response_data.get('error', {}).get('code'), expected_error_code)
    
    def test_non_4d_numpy_array(self):
        all_inputs_np_arr = np.random.rand(5, 6)
        self.run_unsuccessfully_on_np_arrs(all_inputs_np_arr, "request_body_numpy_array_is_not_4d")
    
    def test_non_uint8_numpy_array(self):
        # Test uint16
        all_inputs_np_arr_1 = (np.random.rand(5, 6, 8, 4) * 65535).astype(np.uint16)
        self.run_unsuccessfully_on_np_arrs(all_inputs_np_arr_1, "request_body_numpy_array_is_not_of_type_uint8")

        # Test float
        all_inputs_np_arr_2 = np.random.rand(5, 6, 8, 4).astype(np.float64)
        self.run_unsuccessfully_on_np_arrs(all_inputs_np_arr_2, "request_body_numpy_array_is_not_of_type_uint8")

        # Test bool
        all_inputs_np_arr_3 = (np.random.rand(5, 6, 8, 4) > 0.5).astype(bool)
        self.run_unsuccessfully_on_np_arrs(all_inputs_np_arr_3, "request_body_numpy_array_is_not_of_type_uint8")
    
    def test_non_4_channel_images_numpy_array(self):
        all_inputs_np_arr = np.random.rand(5, 6, 7, 5).astype(np.uint8)
        self.run_unsuccessfully_on_np_arrs(
            all_inputs_np_arr,
            "request_body_numpy_array_is_not_stack_of_4_channel_images"
        )
    
    def run_successfully_on_np_arrs(self, image_np_arrs, draw_outputs=False, output_filename_prefix=""):
        num_inputs = len(image_np_arrs)
        image_height, image_width, _image_cs = image_np_arrs[0].shape

        inputs_np_arr = np.stack(image_np_arrs, axis=0)
        
        request_data_buffer = save_np_arr_to_buffer(inputs_np_arr)

        response = requests_session.post(self.endpoint, data=request_data_buffer.getvalue())
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get('Content-Type', None), 'application/octet-stream')

        masks_arr = load_np_arr_from_stream(response.content)
        self.assertEqual(masks_arr.dtype, np.uint8)
        self.assertEqual(masks_arr.shape, (num_inputs, image_height, image_width))
        
        self.assertIn('X-SEGMENTATION-LABEL-MAP', response.headers)
        try:
            segmentation_label_map = json_loads(response.headers.get('X-SEGMENTATION-LABEL-MAP'))
        except:
            self.assertEqual("segmentation_label_map_is_invalid_json", "segmentation_label_map_is_valid_json")
        
        labels_set = set(i for i in range(len(segmentation_label_map)))
        segmentation_label_values = np.unique(masks_arr)
        for label_value in segmentation_label_values:
            self.assertIn(label_value, labels_set)
        
        if draw_outputs:
            for i, mask in enumerate(masks_arr):
                save_multi_label_segmentation_as_image(
                    path_join(outputs_dir, output_filename_prefix + f"{i}.png"),
                    mask
                )
    
    def test_4d_np_arr_single_input(self):
        num_inputs = 1
        image_height, image_width = randint(100, 1280), randint(100, 720)

        all_images = []
        for _i in range(num_inputs):
            random_image_arr = np.random.randint(0, 255, size=(image_height, image_width, 4), dtype=np.uint8)
            all_images.append(random_image_arr)

        self.run_successfully_on_np_arrs(all_images)
    
    def test_4d_np_arr_multiple_inputs(self):
        num_inputs = randint(2, 10)
        image_height, image_width = randint(100, 1280), randint(100, 720)

        all_images = []
        for _i in range(num_inputs):
            random_image_arr = np.random.randint(0, 255, size=(image_height, image_width, 4), dtype=np.uint8)
            all_images.append(random_image_arr)

        self.run_successfully_on_np_arrs(all_images)

    def test_single_image(self):
        image_arr = cv2.imread(path_join(inputs_dir, "1.jpg"))
        mask_arr = cv2.imread(path_join(inputs_dir, "1-mask.png"), cv2.IMREAD_GRAYSCALE)
        single_input_arr = np.stack(
            [
                image_arr[:, :, 0],
                image_arr[:, :, 1],
                image_arr[:, :, 2],
                mask_arr,
            ],
            axis=-1
        )
        all_images = [single_input_arr]
        
        self.run_successfully_on_np_arrs(all_images, True, "interactive_segment_test_single-")
    
    def test_multiple_images(self):
        image_1_arr = cv2.imread(path_join(inputs_dir, "1.jpg"))
        mask_1_arr = cv2.imread(path_join(inputs_dir, "1-mask.png"), cv2.IMREAD_GRAYSCALE)
        input_1_arr = np.stack(
            [
                image_1_arr[:, :, 0],
                image_1_arr[:, :, 1],
                image_1_arr[:, :, 2],
                mask_1_arr,
            ],
            axis=-1
        )

        image_2_arr = cv2.imread(path_join(inputs_dir, "2.jpg"))
        mask_2_arr = cv2.imread(path_join(inputs_dir, "2-mask.png"), cv2.IMREAD_GRAYSCALE)
        input_2_arr = np.stack(
            [
                image_2_arr[:, :, 0],
                image_2_arr[:, :, 1],
                image_2_arr[:, :, 2],
                mask_2_arr,
            ],
            axis=-1
        )

        all_images = [input_1_arr, input_2_arr]
        
        self.run_successfully_on_np_arrs(all_images, True, "interactive_segment_test_multiple-")


if __name__ == '__main__':
    unittest.main()

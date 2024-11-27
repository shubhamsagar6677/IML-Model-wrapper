import cv2
from cv2 import imread, imwrite, IMREAD_GRAYSCALE
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image  # Import the Pillow library
from json import loads as json_loads
from os import makedirs
from os.path import (
    basename as path_basename,
    dirname as path_dirname,
    isfile as path_isfile,
    join as path_join,
)
from requests import Session
import traceback  # Import traceback for detailed error reporting

from utils import (
    load_np_arr_from_stream,
    multi_label_segmentation_to_colored_image,
    save_np_arr_to_buffer,
)

# Initialize the Flask application
app = Flask(__name__)

# Allow requests from localhost:3000
CORS(app, origins="http://localhost:3000", supports_credentials=True)

# Set up request session and directories
requests_session = Session()
api_base_url = "http://lipikar.cse.iitd.ac.in:7901/api/interactive-model-training"
inputs_dir = path_join(path_dirname(__file__), "inputs")
outputs_dir = path_join(path_dirname(__file__), "outputs")
makedirs(outputs_dir, exist_ok=True)

endpoint = f"{api_base_url}/segment"
endpoint2 = f"{api_base_url}/interactive-segment"

@app.route('/api/segment-image', methods=['POST'])
def segment_image():
    data = request.json
    pixel_data = data.get('pixelData')

    if not pixel_data:
        print("Error: pixelData is required")  # Print the error before returning
        return jsonify({"error": "pixelData is required"}), 400

    try:
        # Check the type and length of pixel_data
        print("Received pixelData type:", type(pixel_data))
        print("Received pixelData length:", len(pixel_data))

        # Ensure that pixel_data has the correct length for reshaping
        image_shape = (512, 512)
        expected_length = image_shape[0] * image_shape[1]

        if len(pixel_data) != expected_length:
            error_message = f"Expected pixelData length {expected_length}, but got {len(pixel_data)}"
            print("Error:", error_message)  # Print the error before returning
            raise ValueError(error_message)

        # Print out the pixel data values for debugging
        print("Pixel Data Sample:", pixel_data[:10])  # Print the first 10 values for inspection

        # Clamp pixel data to the range [0, 255]
        pixel_data_clamped = np.clip(pixel_data, 0, 255)  # Clamp values to [0, 255]

        # Convert the received pixel data to a numpy array and reshape it
        pixel_data_array = np.array(pixel_data_clamped, dtype=np.uint8).reshape(image_shape)
        print("Pixel data array after reshaping:", pixel_data_array)

        # Save the image using Pillow
        image_to_save = Image.fromarray(pixel_data_array)
        image_to_save.save("output_image.png")  # Save the image as a PNG file
        print("Image saved as output_image.png")

        print(pixel_data_array.shape)

        image_arr = cv2.applyColorMap(pixel_data_array, cv2.COLORMAP_JET)
        print(image_arr.shape)
        color_mapped_image_path = path_join(outputs_dir, "color_mapped_segmentation_mask.png")
        cv2.imwrite(color_mapped_image_path, image_arr)


        image_np_arrs = [image_arr]

        num_inputs = len(image_np_arrs)
        image_height, image_width, _image_cs = image_np_arrs[0].shape

        inputs_np_arr = np.stack(image_np_arrs, axis=0)

        request_data_buffer = save_np_arr_to_buffer(inputs_np_arr)

        response = requests_session.post(endpoint, data=request_data_buffer.getvalue())

        # assertEqual(response.status_code, 200)
        # assertEqual(response.headers.get('Content-Type', None), 'application/octet-stream')

        masks_arr = load_np_arr_from_stream(response.content)

        print(masks_arr)
        print(masks_arr.shape)
        masks_arr_reshaped = masks_arr.reshape(512, 512)

        # Save the image in grayscale (assuming the mask is a single channel image)
        output_image_path = path_join(outputs_dir, "segmentation_mask.png")
        cv2.imwrite(output_image_path, masks_arr_reshaped)



        flattened_masks_arr = masks_arr.flatten()

        # Convert it to a list to send in JSON response
        segmentation_data = flattened_masks_arr.tolist()



        # segmentation_image = multi_label_segmentation_to_colored_image(segment_arr)
        # # Make sure to save the image with an appropriate file extension
        # output_file_path = path_join(outputs_dir, "huehue_segment_mask.png")  # Added .png extension
        # imwrite(output_file_path, segmentation_image)
        #
        # # Convert boolean values to integers (0 and 1)
        # segmentation_data = segment_arr.astype(int).flatten().tolist()  # Convert boolean to int and flatten
        # # print(segmentation_data)

        # segmentation_data = [[1 for _ in range(512)] for _ in range(512)]

        # Return the segmentation data as JSON
        return jsonify({"segmentation_data": segmentation_data})


    except Exception as e:
        # Print detailed error message and stack trace
        print("Error:", e)
        print(traceback.format_exc())  # Print the full traceback for debugging
        return jsonify({"error": "An error occurred", "details": str(e)}), 500

@app.route('/api/segment-image-interactive', methods=['POST'])
def segment_image():
    data = request.json
    pixel_data = data.get('pixelData')
    old_seg_data = data.get('old_seg_data')

    if not pixel_data:
        print("Error: pixelData is required")  # Print the error before returning
        return jsonify({"error": "pixelData is required"}), 400

    if not old_seg_data:
        print("Error: old_seg_data is required")  # Print the error before returning
        return jsonify({"error": "old_seg_data is required"}), 400

    try:
        # Check the type and length of pixel_data
        print("Received pixelData type:", type(pixel_data))
        print("Received pixelData length:", len(pixel_data))

        # Ensure that pixel_data has the correct length for reshaping
        image_shape = (512, 512)
        expected_length = image_shape[0] * image_shape[1]

        if len(pixel_data) != expected_length:
            error_message = f"Expected pixelData length {expected_length}, but got {len(pixel_data)}"
            print("Error:", error_message)  # Print the error before returning
            raise ValueError(error_message)

        # Print out the pixel data values for debugging
        print("Pixel Data Sample:", pixel_data[:10])  # Print the first 10 values for inspection

        # Clamp pixel data to the range [0, 255]
        pixel_data_clamped = np.clip(pixel_data, 0, 255)  # Clamp values to [0, 255]

        # Convert the received pixel data to a numpy array and reshape it
        pixel_data_array = np.array(pixel_data_clamped, dtype=np.uint8).reshape(image_shape)
        print("Pixel data array after reshaping:", pixel_data_array)

        # Save the image using Pillow
        image_to_save = Image.fromarray(pixel_data_array)
        image_to_save.save("output_image.png")  # Save the image as a PNG file
        print("Image saved as output_image.png")

        print(pixel_data_array.shape)

        old_seg_data_array = np.array(old_seg_data, dtype=np.uint8).reshape(image_shape)
        mask_arr = old_seg_data_array * 255

        image_arr = cv2.applyColorMap(pixel_data_array, cv2.COLORMAP_JET)
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


        image_np_arrs = [image_arr]

        num_inputs = len(image_np_arrs)
        image_height, image_width, _image_cs = image_np_arrs[0].shape

        inputs_np_arr = np.stack(image_np_arrs, axis=0)

        request_data_buffer = save_np_arr_to_buffer(inputs_np_arr)

        response = requests_session.post(endpoint2, data=request_data_buffer.getvalue())


        masks_arr = load_np_arr_from_stream(response.content)

        print(masks_arr)
        print(masks_arr.shape)
        masks_arr_reshaped = masks_arr.reshape(512, 512)

        # Save the image in grayscale (assuming the mask is a single channel image)
        output_image_path = path_join(outputs_dir, "segmentation_mask.png")
        cv2.imwrite(output_image_path, masks_arr_reshaped)


        flattened_masks_arr = masks_arr.flatten()

        # Convert it to a list to send in JSON response
        segmentation_data = flattened_masks_arr.tolist()

        # Return the segmentation data as JSON
        return jsonify({"segmentation_data": segmentation_data})


    except Exception as e:
        # Print detailed error message and stack trace
        print("Error:", e)
        print(traceback.format_exc())  # Print the full traceback for debugging
        return jsonify({"error": "An error occurred", "details": str(e)}), 500



@app.route('/api/segment-image', methods=['POST'])
def segment_image2():
    # Create a 512x512 array filled with ones
    segmentation_data = np.ones((512, 512), dtype=int).flatten().tolist()  # Flatten the array to a linear list
    print("vfsv ca")

    return jsonify({"segmentation_data": segmentation_data})

if __name__ == '__main__':
    app.run(port=8000)

# Flask Data Augmentation API

This Flask-based API allows you to augment a dataset of images in YOLO format with bounding boxes drawn on them. The augmentations are performed using the `albumentations` library, and the API returns a zip file containing the augmented images with bounding boxes.

## Features
- Apply various augmentations such as crop, blur, rotate, horizontal flip, and vertical flip.
- Specify probabilities for each augmentation.
- Draw bounding boxes on augmented images.
- Return augmented images as a zip file.

## Requirements
- Python 3.x
- Flask
- Pillow
- Albumentations
- Numpy

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/Atulsah17/Flask-Data-Augmentation-API.git

    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask application:

    ```bash
    python app.py
    ```

The API will now be running locally at `http://127.0.0.1:5000`.

## API Endpoint

### `/augment` - POST

This endpoint accepts a dataset in YOLO format and applies augmentations based on the specified parameters. It returns a zip file containing augmented images with bounding boxes drawn on them.

#### Request

- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### Form Data Parameters:
1. **`num_images`** *(required)*: The number of images to generate (integer).
2. **`crop_width`**: Width of the crop (default: 256).
3. **`crop_height`**: Height of the crop (default: 256).
4. **`crop_prob`**: Probability of applying crop augmentation (default: 0.5).
5. **`blur_prob`**: Probability of applying blur augmentation (default: 0.2).
6. **`rotate_limit`**: Max angle for rotation (default: 30 degrees).
7. **`rotate_prob`**: Probability of applying rotation (default: 0.3).
8. **`horizontal_flip_prob`**: Probability of applying horizontal flip (default: 0.5).
9. **`vertical_flip_prob`**: Probability of applying vertical flip (default: 0.5).
10. **`dataset`** *(required)*: The original dataset in YOLO format as a zip file.

#### Example Request:
```bash
curl -X POST http://127.0.0.1:5000/augment \
-F "num_images=10" \
-F "crop_width=300" \
-F "crop_height=300" \
-F "crop_prob=0.6" \
-F "rotate_limit=40" \
-F "rotate_prob=0.4" \
-F "horizontal_flip_prob=0.5" \
-F "vertical_flip_prob=0.5" \
-F "dataset=@/path/to/your/yolo_dataset.zip"

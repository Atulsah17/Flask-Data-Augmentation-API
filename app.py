import os
import zipfile
import io
import random
from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageDraw
import numpy as np
import albumentations as A

app = Flask(__name__)

def get_augmentation_pipeline(augmentations):
    """
    Creates an augmentation pipeline using Albumentations library based on the provided augmentations.
    
    Args:
    augmentations (dict): A dictionary of augmentation operations with parameters like probability.
    
    Returns:
    A.Compose: A composed Albumentations pipeline.
    """
    return A.Compose([
        A.RandomCrop(width=augmentations['crop']['width'], height=augmentations['crop']['height'], p=augmentations['crop']['probability']),
        A.Blur(blur_limit=3, p=augmentations['blur']['probability']),
        A.Rotate(limit=augmentations['rotate']['limit'], p=augmentations['rotate']['probability']),
        A.HorizontalFlip(p=augmentations['horizontal_flip']['probability']),
        A.VerticalFlip(p=augmentations['vertical_flip']['probability'])
    ])

def yolo_to_bbox(yolo_bbox, img_width, img_height):
    """
    Converts YOLO format bounding box to standard (x1, y1, x2, y2) bounding box format.
    
    Args:
    yolo_bbox (list): Bounding box in YOLO format (class_id, x_center, y_center, width, height).
    img_width (int): Width of the image.
    img_height (int): Height of the image.
    
    Returns:
    tuple: Standard bounding box format (x1, y1, x2, y2).
    """
    class_id, x_center, y_center, width, height = yolo_bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return x1, y1, x2, y2

def augment_image(image_path, label_path, augmentations):
    """
    Applies augmentations to the image and draws bounding boxes on the augmented image.
    
    Args:
    image_path (str): Path to the image.
    label_path (str): Path to the YOLO format label file.
    augmentations (dict): Dictionary of augmentation parameters.
    
    Returns:
    Image: Augmented image with bounding boxes drawn on it.
    """
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    img_width, img_height = image.size

    # Extract bounding boxes from the label file
    bboxes = []
    if os.path.isfile(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                yolo_bbox = list(map(float, line.strip().split()))
                if len(yolo_bbox) == 5:  # YOLO format has 5 elements
                    bboxes.append(yolo_to_bbox(yolo_bbox, img_width, img_height))

    # Apply augmentations to the image
    augment_pipeline = get_augmentation_pipeline(augmentations)
    augmented = augment_pipeline(image=image_np)
    augmented_image = Image.fromarray(augmented['image'])

    # Draw bounding boxes on the augmented image
    draw = ImageDraw.Draw(augmented_image)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    return augmented_image

@app.route('/augment', methods=['POST'])
def augment_endpoint():
    """
    Flask endpoint to handle dataset augmentation. It extracts the dataset,
    applies augmentations, draws bounding boxes on the images, and returns a ZIP file with augmented images.
    
    Returns:
    Flask response: A ZIP file containing augmented images with bounding boxes.
    """
    try:
        # Create temporary directories for extracting and storing augmented images
        temp_dir = 'temp'
        augmented_dir = 'augmented'
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(augmented_dir, exist_ok=True)

        # Get the number of images to augment
        num_images_str = request.form.get('num_images', '').strip()
        if not num_images_str.isdigit():
            return jsonify({'error': 'Invalid number of images'}), 400
        num_images = int(num_images_str)

        # Parse the augmentations from the request
        augmentations = {
            'crop': {
                'width': int(request.form.get('crop_width', 256)),
                'height': int(request.form.get('crop_height', 256)),
                'probability': float(request.form.get('crop_prob', 0.5))
            },
            'blur': {
                'probability': float(request.form.get('blur_prob', 0.2))
            },
            'rotate': {
                'limit': int(request.form.get('rotate_limit', 30)),
                'probability': float(request.form.get('rotate_prob', 0.3))
            },
            'horizontal_flip': {
                'probability': float(request.form.get('horizontal_flip_prob', 0.5))
            },
            'vertical_flip': {
                'probability': float(request.form.get('vertical_flip_prob', 0.5))
            }
        }

        # Check if dataset zip file is provided
        if 'dataset' not in request.files:
            return jsonify({'error': 'No dataset provided'}), 400

        dataset_zip = request.files['dataset']
        dataset_path = os.path.join(temp_dir, 'dataset.zip')
        dataset_zip.save(dataset_path)

        # Extract the dataset from the zip file
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Set paths for images and labels
        images_dir = os.path.join(temp_dir, 'dataset', 'images')
        labels_dir = os.path.join(temp_dir, 'dataset', 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            return jsonify({'error': 'Invalid dataset structure. Ensure "images" and "labels" directories are present.'}), 400

        # List image files and select a random subset
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        selected_images = random.sample(image_files, min(num_images, len(image_files)))

        # Create a zip file to hold the augmented images
        output_zip_path = os.path.join(augmented_dir, 'augmented_images.zip')
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            for image_file in selected_images:
                image_path = os.path.join(images_dir, image_file)
                label_file = image_file.rsplit('.', 1)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)

                # If the label file is missing, skip this image
                if not os.path.isfile(label_path):
                    print(f"Warning: Skipping image due to missing label file: {label_file}")
                    continue

                # Augment the image and draw bounding boxes
                augmented_image = augment_image(image_path, label_path, augmentations)
                augmented_image_name = f"{os.path.splitext(image_file)[0]}_aug.jpg"

                # Save the augmented image to the zip file
                img_byte_arr = io.BytesIO()
                augmented_image.save(img_byte_arr, format="JPEG")
                zipf.writestr(augmented_image_name, img_byte_arr.getvalue())

        # Return the zip file containing the augmented images
        return send_file(output_zip_path, download_name='augmented_images.zip', as_attachment=True)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

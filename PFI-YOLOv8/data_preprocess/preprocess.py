import os
import json

# Directory paths
current_directory = r"D:\Code\Python\PFI-YOLOv8\datasets\T1、T2 PFI AI 辅助\8 Junior2+AI-PFI-T2"
json_directory = os.path.join(current_directory, "json")
txt_directory = os.path.join(current_directory, "labels")

# Ensure the txt directory exists
os.makedirs(txt_directory, exist_ok=True)

# Define label mapping
label_mapping = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


# Function to normalize coordinates
def normalize_coordinates(points, image_width, image_height):
    normalized_points = []
    for point in points:
        x = round(point[0] / image_width,4)
        y = round(point[1] / image_height,4)
        normalized_points.extend([x, y])
    return normalized_points


# Process each JSON file
for json_file in os.listdir(json_directory):
    if json_file.endswith('.json'):
        json_path = os.path.join(json_directory, json_file)

        # Read JSON file
        with open(json_path, 'r',encoding='utf-8') as file:
            data = json.load(file)

        # Get image dimensions
        image_width = data['imageWidth']
        image_height = data['imageHeight']

        # Process shapes
        yolo_annotations = []
        for shape in data['shapes']:
            label = shape['label']
            if label in label_mapping:
                class_index = label_mapping[label]
                points = shape['points']
                normalized_points = normalize_coordinates(points, image_width, image_height)
                yolo_annotation = [class_index] + normalized_points
                yolo_annotations.append(yolo_annotation)

        # Write to txt file
        txt_file_name = os.path.splitext(json_file)[0] + '.txt'
        txt_path = os.path.join(txt_directory, txt_file_name)

        with open(txt_path, 'w') as file:
            for annotation in yolo_annotations:
                annotation_str = ' '.join(map(str, annotation))
                file.write(annotation_str + '\n')

print("Conversion completed.")

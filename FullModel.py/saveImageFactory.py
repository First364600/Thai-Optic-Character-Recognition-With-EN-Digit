import cv2
import os
import datetime

# datetime.datetime.fo
# print(format)

def saveImage(image, base_filename, output_dir):
    time = datetime.datetime.now()
    format = time.strftime("%d%m%y_%H%M%S")
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the initial file path
    file_path = os.path.join(output_dir, base_filename)

    # Get the file extension and base name
    base_name, ext = os.path.splitext(file_path)
    
    # Initialize a counter for duplicate filenames
    counter = 1

    # Check if the file already exists, and if so, create a new unique filename
    file_path = f"{base_name[:-1]}{format}{ext}"
    # while os.path.exists(file_path):
    #     counter += 1

    # Save the image using OpenCV
    # print(file_path)
    cv2.imwrite(file_path, image)

# Example usage:
# Load an example image (replace with your image)
# image = cv2.imread('example_image.jpg')  # Replace with your image path

# # Save the image with a unique name in the specified directory
# output_dir = 'output_images'  # Replace with your output directory
# unique_filename = save_image_with_unique_name(image, 'saved_image.png', output_dir)

# print(f"Image saved as {unique_filename}")

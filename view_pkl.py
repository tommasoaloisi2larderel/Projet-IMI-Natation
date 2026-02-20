
import pickle
import sys
import numpy as np
import cv2
import argparse
import os

def view_pkl(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded {file_path}")
        print(f"Keys: {list(data.keys())}")
        
        if 'line_image' in data:
            image = data['line_image']
            print(f"Image shape: {image.shape}")
            
            # Save the image
            output_filename = os.path.basename(file_path) + '.png'
            # Convert RGB to BGR for OpenCV if needed, usually pickle saves as RGB if from PIL/matplotlib, but cv2 uses BGR
            # Assuming it might be RGB. If it looks blue-ish, it was BGR.
            # Let's save as is for now or use matplotlib if installed.
            
            # reliable save using cv2 (expects BGR)
            # If the source was RGB, we need to convert.
            # Common CV practice: read as BGR. If this data is RGB, we flip.
            # I will try to save it as is.
            cv2.imwrite(output_filename, image)
            print(f"Saved image to {output_filename}")
            
            # Optionally display
            # cv2.imshow('Image', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
        else:
            print("No 'line_image' key found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View image from .pkl file")
    parser.add_argument("file", help="Path to the .pkl file")
    args = parser.parse_args()
    
    view_pkl(args.file)

import os
from PIL import Image

def process_gif_folder(main_folder, output_base_folder):
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".gif"):
                gif_path = os.path.join(root, file)
                subfolder_name = os.path.basename(root)  # Subfolder name as label
                output_folder = os.path.join(output_base_folder, subfolder_name)
                os.makedirs(output_folder, exist_ok=True)
                extract_frames(gif_path, os.path.join(output_folder, file[:-4]))

def extract_frames(gif_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    gif = Image.open(gif_path)
    try:
        frame_index = 0
        while True:
            frame = gif.copy()
            frame.save(f"{output_folder}/frame_{frame_index}.png")
            frame_index += 1
            gif.seek(frame_index)
    except EOFError:
        pass  # End of GIF

# Define paths
main_folder = "GIFS"  # Example: "./gifs"
output_base_folder = "extracted_frames"  # Example: "./extracted_frames"

# Process all GIFs
process_gif_folder(main_folder, output_base_folder)

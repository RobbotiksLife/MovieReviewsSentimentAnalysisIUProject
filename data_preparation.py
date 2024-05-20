import os
import glob
from tqdm import tqdm

def read_txt_files(folder_path):
    # Create an empty dictionary to store file names and contents
    files_dict = {}

    # Get all .txt files in the specified folder
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

    # Iterate over each .txt file
    for file_path in tqdm(txt_files, desc=f"Reading files from {folder_path}"):
        # Get the file name without the directory path
        file_name = os.path.basename(file_path)

        # Open the file and read its contents
        with open(file_path, 'r', encoding='utf-8') as file:
            file_contents = file.read()

        # Add the file name and contents to the dictionary
        files_dict[file_name] = file_contents

    return files_dict
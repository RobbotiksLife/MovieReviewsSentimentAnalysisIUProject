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


def define_dataset(main_dataset_folder_path = "aclImdb/"):
    # Load train data
    train_neg = read_txt_files(folder_path=f"{main_dataset_folder_path}train/neg")
    train_pos = read_txt_files(folder_path=f"{main_dataset_folder_path}train/pos")
    # Load test data
    test_neg = read_txt_files(folder_path=f"{main_dataset_folder_path}test/neg")
    test_pos = read_txt_files(folder_path=f"{main_dataset_folder_path}test/pos")

    # Create lists of reviews and labels
    train_reviews = list(train_neg.values()) + list(train_pos.values())
    train_labels = [0] * len(train_neg) + [1] * len(train_pos)
    test_reviews = list(test_neg.values()) + list(test_pos.values())
    test_labels = [0] * len(test_neg) + [1] * len(test_pos)

    return train_reviews, train_labels, test_reviews, test_labels
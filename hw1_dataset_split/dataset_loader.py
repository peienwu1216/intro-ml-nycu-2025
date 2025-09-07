import os
import random

def load_and_split_dataset(base_dir, train_ratio=0.7, val_ratio=0.15):
    
    all_files_by_label = {}
    
    # 1. Loading all files from data / sort by labels
    print(f"Loading data from: {os.path.abspath(base_dir)}")
    if not os.path.exists(base_dir):
        print(f"Error: Dataset directory not found at '{os.path.abspath(base_dir)}'")
        return [], [], []

    for label_folder in sorted(os.listdir(base_dir)):
        label_path = os.path.join(base_dir, label_folder)
        if os.path.isdir(label_path):
            try:
                digit = int(label_folder.split('_')[1])
                images = [os.path.join(label_folder, f) for f in sorted(os.listdir(label_path))]
                all_files_by_label[digit] = images
            except (ValueError, IndexError):
                continue

    train_list = []
    val_list = []
    test_list = []

    # 2. Catogorize images by digits label
    for digit, images in all_files_by_label.items():
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        class_train = images[:n_train]
        class_val = images[n_train : n_train + n_val]
        class_test = images[n_train + n_val :]

        train_list.extend([(img_path, digit) for img_path in class_train])
        val_list.extend([(img_path, digit) for img_path in class_val])
        test_list.extend([(img_path, digit) for img_path in class_test])

    # 3. Write in file
    def save_split(filepath, data_list):
        with open(filepath, 'w') as f:
            for img_path, label in data_list:
                f.write(f"{img_path.replace(os.sep, '/')} {label}\n")

    save_split('train_list.txt', train_list)
    save_split('val_list.txt', val_list)
    save_split('test_list.txt', test_list)

    # 4. Output results in terminal
    print("\nDataset splitting complete.")
    print(f"Train set size: {len(train_list)}")
    print(f"Validation set size: {len(val_list)}")
    print(f"Test set size: {len(test_list)}")
    
    return train_list, val_list, test_list

def check_data_leakage(train_set, val_set, test_set):

    print("\n--- Checking for Data Leakage ---")
    
    if not all([train_set, val_set, test_set]):
        print("One or more sets are empty. Skipping leakage check.")
        return True

    train_paths = {item[0] for item in train_set}
    val_paths = {item[0] for item in val_set}
    test_paths = {item[0] for item in test_set}

    leakage_train_val = train_paths.intersection(val_paths)
    leakage_train_test = train_paths.intersection(test_paths)
    leakage_val_test = val_paths.intersection(test_paths)

    has_leakage = False
    if leakage_train_val:
        print(f"Error: Leakage detected between training and validation sets: {len(leakage_train_val)} samples.")
        has_leakage = True
    if leakage_train_test:
        print(f"Error: Leakage detected between training and test sets: {len(leakage_train_test)} samples.")
        has_leakage = True
    if leakage_val_test:
        print(f"Error: Leakage detected between validation and test sets: {len(leakage_val_test)} samples.")
        has_leakage = True
    
    if not has_leakage:
        print("Success: No data leakage detected. The splits are mutually exclusive.")
        return True
    
    return False

if __name__ == '__main__':

    DATASET_ROOT = "../data/MNIST"
    # Load and split data set
    train_data, val_data, test_data = load_and_split_dataset(base_dir=DATASET_ROOT)
    
    # Bonus: Check data leakage
    check_data_leakage(train_data, val_data, test_data)
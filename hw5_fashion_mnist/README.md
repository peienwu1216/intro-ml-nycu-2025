# HW5: Build & Compare NN vs CNN (Image Classification)
Student ID: 113511103

## 1. Project Structure
.
├── main.py                 # Main training script (generates pred_tta.csv)
├── hw5_fashion_mnist.ipynb # Notebook for ablation studies and visualization
└── README.txt              # This file

## 2. Environment Requirements
- Python 3.x
- PyTorch & Torchvision
- Pandas
- NumPy
- Matplotlib
- Tqdm
- Scikit-learn

## 3. Data Setup
The code expects the dataset files to be located in a `data/` folder relative to the script.
1. Create a folder named `data/`.
2. Download `train.csv` and `test4students.csv` from Kaggle.
3. Place them inside the `data/` folder.

File structure should look like:
code/
  ├── data/
  │   ├── train.csv
  │   └── test4students.csv
  ├── main.py
  └── hw5_fashion_mnist.ipynb

## 4. How to Run
To train the final Improved CNN model and generate the prediction file:

python main.py

- The script sets the random seed to 58 for reproducibility.
- It will train for 60 epochs (with early stopping).
- After training, it generates `pred_tta.csv` for Kaggle submission.
- For visualization and ablation test, look for 'hw5_fashion_mnist.ipynb'

## 5. Notes
- The model uses MPS (Mac) or CUDA automatically if available.
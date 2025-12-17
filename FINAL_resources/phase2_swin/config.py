import os

class Config:
    # Paths
    # Assuming we are running from the root of the workspace or field directory
    # Adjust these paths based on where the script is executed
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), 'CADB_Dataset')
    
    # Training Settings
    BATCH_SIZE = 16 # Reduced for Swin + 384x384
    NUM_WORKERS = 4
    MAX_EPOCH = 50
    LR = 1e-4
    WEIGHT_DECAY = 5e-5
    MOMENTUM = 0.9
    OPTIMIZER = 'adamw'
    
    # Model Settings
    IMAGE_SIZE = 384
    BACKBONE = 'swin_t' # swin_t, swin_s, swin_b
    DROPOUT = 0.5
    
    # Loss Weights
    LAMBDA_EMD = 1.0
    LAMBDA_ATTR = 0.1
    LAMBDA_RANK = 0.2 # New Ranking Loss weight
    
    # Attributes
    ATTRIBUTE_TYPES = ['RuleOfThirds', 'BalacingElements', 'DoF', 'Object', 'Symmetry', 'Repetition']
    NUM_ATTRIBUTES = len(ATTRIBUTE_TYPES)
    
    # Experiment
    EXP_NAME = 'Swin_SpatialAtt_Rank'
    LOG_DIR = os.path.join(BASE_DIR, 'logs', EXP_NAME)
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints', EXP_NAME)
    
    @staticmethod
    def create_dirs():
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

if __name__ == '__main__':
    Config.create_dirs()

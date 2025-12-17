import os

class Config:
    # ==========================================
    # [Control Variable] Paths
    # ==========================================
    # Assuming we are running from the root of the workspace or field directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Path to the shared CADB Dataset. Do NOT change this if you want to use the same data split.
    DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), 'CADB_Dataset')
    
    # ==========================================
    # [Independent Variable] Training Settings
    # ==========================================
    # You can tune these for your specific model
    BATCH_SIZE = 16 
    NUM_WORKERS = 4
    MAX_EPOCH = 50
    
    # Optimizer settings
    LR_BACKBONE = 1e-5
    LR_HEAD = 2e-4
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    OPTIMIZER = 'adamw'
    
    # ==========================================
    # [Independent Variable] Model Settings
    # ==========================================
    IMAGE_SIZE = 384 # Recommended to keep 384 for fair comparison with current SOTA
    BACKBONE = 'convnextv2_nano' 
    DROPOUT = 0.5
    
    # ==========================================
    # [Control Variable] Loss Weights
    # ==========================================
    # These weights define the optimization objective. 
    # Changing them changes the problem definition slightly.
    LAMBDA_EMD = 1.0
    LAMBDA_ATTR = 0.1
    LAMBDA_RANK = 0.2 
    
    # ==========================================
    # [Control Variable] Attributes
    # ==========================================
    ATTRIBUTE_TYPES = ['RuleOfThirds', 'BalacingElements', 'DoF', 'Object', 'Symmetry', 'Repetition']
    NUM_ATTRIBUTES = len(ATTRIBUTE_TYPES)
    
    # ==========================================
    # [Experiment Info]
    # ==========================================
    EXP_NAME = 'My_New_Framework' # Change this for your new experiment
    LOG_DIR = os.path.join(BASE_DIR, 'logs', EXP_NAME)
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints', EXP_NAME)
    
    @staticmethod
    def create_dirs():
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

if __name__ == '__main__':
    Config.create_dirs()

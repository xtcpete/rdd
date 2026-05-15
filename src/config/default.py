from yacs.config import CfgNode as CN
_CN = CN()

##############  ↓  RDD Pipeline  ↓  ##############
_CN.RDD = CN()
_CN.RDD.TOP_K = 4096
_CN.RDD.DETECTION_THR = 0.1
_CN.RDD.TRAIN_DETECTOR = False

# 1. descriptor
_CN.RDD.DESCRIPTOR = CN()
_CN.RDD.DESCRIPTOR.BACKBONE = 'convnext_tiny'
_CN.RDD.DESCRIPTOR.USE_22K = True
_CN.RDD.DESCRIPTOR.D_MODEL = 256
_CN.RDD.DESCRIPTOR.NUM_ENCODER_LAYERS = 4
_CN.RDD.DESCRIPTOR.NHEADS = 8
_CN.RDD.DESCRIPTOR.DROPOUT = 0.1
_CN.RDD.DESCRIPTOR.ACTIVATION = 'gelu'  # ['relu', 'gelu']
_CN.RDD.DESCRIPTOR.DIM_FEEDFORWARD = 1024
_CN.RDD.DESCRIPTOR.NUM_FEATURE_LEVELS = 5
_CN.RDD.DESCRIPTOR.ENC_N_POINTS = 8
_CN.RDD.DESCRIPTOR.USE_DEFORMABLE_TRANSFORMER = True
_CN.RDD.DESCRIPTOR.PRETRAIN_BACKBONE = True

# 2. detector
_CN.RDD.DETECTOR = CN()
_CN.RDD.DETECTOR.TYPE = 'legacy'
_CN.RDD.DETECTOR.BLOCK_DIMS = [8, 16, 32, 64]
_CN.RDD.DETECTOR.INPUT_DIM = 256
_CN.RDD.DETECTOR.UPSAMPLE_DIMS = [128, 64]
_CN.RDD.DETECTOR.JOINT_TRAINING = False

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()

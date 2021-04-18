import tensorflow as tf
from loss_funnction_And_matrics import*
import numpy as np
np.random.seed(42)

####---Input-Data---###
TRAIN_CSV="/image_data/Scripts/NLP_Classification_CNN/Kidney_Classification/Kiney_Train_csv.csv"
VAL_CSV="/image_data/Scripts/NLP_Classification_CNN/Kidney_Classification/Kiney_Val_csv.csv"
REPORT_TEXT_COLUMN_NAME='text_Finding_only'
LABELS_COLUMN_NAMES=['Kidney_stone_lbl','Kidney_atroph_lbl','Kidney_lesion_lbl','Kidney_cyst_lbl','Kidney_normal_lbl']

#=-----Model Configuration----###
NUMBER_OF_CLASSES=5
MAX_WORDS=650
EMBADDING_DIMENTION=200
USING_PRE_TRAINED_EMBADDING=True # (True/False) if True: Will use Pretrained Embading
PRE_TRAINING_EMBADDING="/image_data/Scripts/BioWordVec_PubMed_MIMICIII_d200.vec.bin.gz"

###----Resume-Training----###
RESUME_TRAINING=0
RESUME_TRAIING_MODEL='/image_data/Scripts/NLP_Classification_CNN/Kidney_Classification/Multi_Label_Pretrained/Model_Multi_Label_Pretrained/'
TRAINING_INITIAL_EPOCH=0

##--------Training-----Hyperperametter---------
TRAING_EPOCH=50
TRAIN_CLASSIFY_LEARNING_RATE =1e-4
TRAIN_CLASSIFY_LOSS=Weighted_BCTL #tf.keras.losses.BinaryCrossentropy() #Weighted_BCTL
OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)
TRAIN_CLASSIFY_METRICS=tf.keras.metrics.AUC()
BATCH_SIZE=512
SHUFFLE=True
###-------SAVING_UTILITY-----########
ModelCheckpoint_MOTITOR='val_loss'
TRAINING_SAVE_MODEL_PATH='/image_data/Scripts/NLP_Classification_CNN/Kidney_Classification/Multi_Label_Pretrained/Model_Multi_Label_Pretrained/'
TRAINING_CSV='Kidney_Multi_Label_Pretrained.csv'
LOG_NAME="Log_Kidney_Multi_Label_Pretrained"
MODEL_SAVING_NAME="nlp_Kidney_Multi_Label_Pretrained{val_loss:.2f}_{epoch}.h5"

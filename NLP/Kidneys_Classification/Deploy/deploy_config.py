import tensorflow as tf
from loss_funnction_And_matrics import*
import numpy as np
np.random.seed(42)


PUNCTUATION=['"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', 'â€¢',  '~', '@', 'Â£',
     'Â·', '_', '{', '}', 'Â©', '^', 'Â®', '`',  '<', 'â†’', 'Â°', 'â‚¬', 'â„¢', 'â€º',  'â™¥', 'â†', 'Ã—', 'Â§', 'â€³', 'â€²', 'Ã‚', 'â–ˆ', 'Â½', 'Ã ', 'â€¦',
     'â€œ', 'â˜…', 'â€', 'â€“', 'â—', 'Ã¢', 'â–º', 'âˆ’', 'Â¢', 'Â²', 'Â¬', 'â–‘', 'Â¶', 'â†‘', 'Â±', 'Â¿', 'â–¾', 'â•', 'Â¦', 'â•‘', 'â€•', 'Â¥', 'â–“', 'â€”', 'â€¹', 'â”€',
     'â–’', 'ï¼š', 'Â¼', 'âŠ•', 'â–¼', 'â–ª', 'â€ ', 'â– ', 'â€™', 'â–€', 'Â¨', 'â–„', 'â™«', 'â˜†', 'Ã©', 'Â¯', 'â™¦', 'Â¤', 'â–²', 'Ã¨', 'Â¸', 'Â¾', 'Ãƒ', 'â‹…', 'â€˜', 'âˆž',
     'âˆ™', 'ï¼‰', 'â†“', 'ã€', 'â”‚', 'ï¼ˆ', 'Â»', 'ï¼Œ', 'â™ª', 'â•©', 'â•š', 'Â³', 'ãƒ»', 'â•¦', 'â•£', 'â•”', 'â•—', 'â–¬', 'â¤', 'Ã¯', 'Ã˜', 'Â¹', 'â‰¤', 'â€¡', 'âˆš', ]

#=-----Model Configuration----###
NUMBER_OF_CLASSES=5
MAX_WORDS=650
EMBADDING_DIMENTION=200
USING_PRE_TRAINED_EMBADDING=True # (True/False) if True: Will use Pretrained Embading
PRE_TRAINING_EMBADDING="/image_data/Scripts/BioWordVec_PubMed_MIMICIII_d200.vec.bin.gz"
##---Training--Hyperperametter---###
TRAIN_CLASSIFY_LEARNING_RATE =1e-4
TRAIN_CLASSIFY_LOSS=Weighted_BCTL #tf.keras.losses.BinaryCrossentropy() #Weighted_BCTL
OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)
TRAIN_CLASSIFY_METRICS=tf.keras.metrics.AUC()


####---Input-Data---###---Kidneys
TEST_CSV="/image_data/Scripts/NLP_Classification_CNN/Kiney_Manual_Test_set_patient.csv"



REPORT_TEXT_COLUMN_NAME='text_Finding_only'
#[#'text_finding_impression_list'#'text_impression_only'#'text_Finding_only']
SUBJECT_ID_COLUMN_NAME='Report_id'
LABELS_COLUMN_NAMES=['Kidney_stone_lbl','Kidney_atroph_lbl','Kidney_lesion_lbl','Kidney_cyst_lbl','Kidney_normal_lbl']
LABELS_COLUMB_BINARY_LABEL_NAME=['Kidney_normal_lbl']

TOKENIZER_PICKLE="tokenizer.pickle"
MODEL_WIGHT='nlp_Kidney_Multi_Label_Pretrained.h5'
SAVING_CSV="nlp_Kidney_Multi_Label_Pretrained_Test_ManualTest_HTML.csv"
SAVING_HTML_FILE='path/to/save/html/Kidneys_MultiLabel_Pretrained_ManualTestset_HTML/'

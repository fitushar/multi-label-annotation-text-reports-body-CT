import tensorflow as tf
from loss_funnction_And_matrics import*
import numpy as np
np.random.seed(42)

PUNCTUATION=['"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
     '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
     '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
     '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
     '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

#=-----Model Configuration----###
NUMBER_OF_CLASSES=5
MAX_WORDS=650
EMBADDING_DIMENTION=200
USING_PRE_TRAINED_EMBADDING=True # (True/False) if True: Will use Pretrained Embading
PRE_TRAINING_EMBADDING="/path/to/BioWordVec_PubMed_MIMICIII_d200.vec.bin.gz"
##---Training--Hyperperametter---###
TRAIN_CLASSIFY_LEARNING_RATE =1e-4
TRAIN_CLASSIFY_LOSS=Weighted_BCTL #tf.keras.losses.BinaryCrossentropy() #Weighted_BCTL
OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)
TRAIN_CLASSIFY_METRICS=tf.keras.metrics.AUC()


####---Input-Data---###---Livers
TEST_CSV="/path/to/Liver_Manual_Test_set_patient.csv"


REPORT_TEXT_COLUMN_NAME='text_Finding_only' #'text_Finding_only' #'text_impression_only' #'text_Finding_only'
SUBJECT_ID_COLUMN_NAME='Report_id'
LABELS_COLUMN_NAMES=['liver_stone_lbl','liver_lesion_lbl','liver_dilat_lbl','liver_fatty_liver_lbl','liver_normal_lbl']
LABELS_COLUMB_BINARY_LABEL_NAME=['liver_normal_lbl']


#-----Multi-label
TOKENIZER_PICKLE="tokenizer.pickle"
MODEL_WIGHT='nlp_Liver_Multi_Label_wight.h5'
SAVING_CSV="nlp_Liver_Multi_label_Test_ManualTest_HTML.csv"
SAVING_HTML_FILE='/path/to/save/html/Kidneys_MultiLabel_Pretrained_ManualTestset_HTML/'

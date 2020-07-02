#-------------------------------
# File Path
#-------------------------------
PROJECT_PATH = "<Path_To_Main_Package>"
DATASET_DIR = "<Path_To_Yelp_Dataset>"

DATA_DIR = PROJECT_PATH + "data/"
OUTPUT_DIR = PROJECT_PATH + "output/"

BUSINESS = DATASET_DIR + "yelp_academic_dataset_business.json"
REVIEW = DATASET_DIR + "yelp_academic_dataset_review.json"

#-------------------------------
# Constants
#-------------------------------
COL_USER = "int_user_id"
COL_ITEM = "feature_id"
COL_RATING = "rating"
COL_PREDICTION = "prediction"
COL_TIMESTAMP = "timestamp"

FEATURE_SEPARATOR = "::"
KEY_VAL_SEPARATOR = ";;"
COL_SEPARATOR = "\t"
DEFAULT_VAL_IF_NOT_EXIST = "!other"

# feature_vector_to_int in json format
FEATURE_ID = "feature_id"
FEATURE_DICT = "feature_dict"

TOP_K = 10
THRESHOLD_SCALE_5 = 3
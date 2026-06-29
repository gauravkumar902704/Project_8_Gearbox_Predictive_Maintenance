import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

HEALTHY_FOLDER = os.path.join(BASE_DIR,"Dataset","Healthy Data")
BROKEN_FOLDER = os.path.join(BASE_DIR,"Dataset","BrokenTooth Data")

MODEL_PATH = os.path.join(BASE_DIR,"Models","Gearbox_model.pkl")
CSV_PATH = os.path.join(BASE_DIR,"Outputs","Gearbox_features.csv")
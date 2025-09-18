import kagglehub
import os
import shutil
import pathlib
from transformers import AutoTokenizer
import datasets
import torch

CLASSES = [
    "Computer Science",
    "Physics",
    "Mathematics",
    "Statistics",
    "Quantitative Biology",
    "Quantitative Finance"
]
CLASS2ID = {class_:int(id) for id, class_ in enumerate(CLASSES)}
ID2CLASS = {int(id):class_ for class_, id in CLASS2ID.items()}
TOKENIZER = AutoTokenizer.from_pretrained("microsoft/deberta-base")

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
REL_DATASET_DIR = "./dataset/"

os.chdir(CURRENT_DIR)

# Import dataset

def import_dataset():

    # Download latest version
    path = kagglehub.dataset_download("shivanandmn/multilabel-classification-dataset")

    shutil.copytree(path, REL_DATASET_DIR, dirs_exist_ok=True)
    shutil.rmtree(path)

def preprocess_train(entry):
    text = f"{entry["TITLE"]}:\n{entry["ABSTRACT"]}"
    labels = torch.tensor([entry[class_] for class_ in CLASSES]).float()
    entry = TOKENIZER(text, truncation=True)
    entry['labels'] = labels
    return entry

def preprocess_test(entry):
    text = f"{entry["TITLE"]}:\n{entry["ABSTRACT"]}"
    entry = TOKENIZER(text, truncation=True)
    return entry

if not os.path.exists(REL_DATASET_DIR):
    import_dataset()

if not os.path.exists(f"{REL_DATASET_DIR}train" or os.path.exists(f"{REL_DATASET_DIR}test")):
    TRAIN_DATASET = datasets\
        .load_dataset("csv", data_files=f"{REL_DATASET_DIR}train.csv")\
        .map(preprocess_train)["train"]\
        .train_test_split(test_size=.1, seed=0)
    
    TEST_DATASET = datasets\
        .load_dataset("csv", data_files=f"{REL_DATASET_DIR}test.csv")\
        .map(preprocess_test)
        
    TRAIN_DATASET.save_to_disk(f"{REL_DATASET_DIR}train")
    TEST_DATASET.save_to_disk(f"{REL_DATASET_DIR}test")

else:
    TRAIN_DATASET = datasets.load_from_disk(f"{REL_DATASET_DIR}train")
    TEST_DATASET = datasets.load_from_disk(f"{REL_DATASET_DIR}test")
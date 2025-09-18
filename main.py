from training import trainer
from data import CURRENT_DIR, CLASSES, ID2CLASS, CLASS2ID
from transformers import AutoModelForSequenceClassification
import os

os.chdir(CURRENT_DIR)

if __name__ == "__main__":
    if not os.path.exists("./model/"):
        trainer.train()
    
    trained_model = AutoModelForSequenceClassification.from_pretrained(
        "./model",
        num_labels=len(CLASSES),
        id2label=ID2CLASS,
        label2id=CLASS2ID,
        problem_type = "multi_label_classification"
    )
    # TODO: Add inference over testing dataset
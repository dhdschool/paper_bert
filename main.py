from training import trainer
from data import CURRENT_DIR, CLASSES, ID2CLASS, CLASS2ID, TEST_DATASET
from transformers import AutoModelForSequenceClassification
import os
import torch
import pandas as pd

os.chdir(CURRENT_DIR)

if __name__ == "__main__":
    # Train model
    if not os.path.exists("./model/"):
        trainer.train()
    
    # Load trained model
    trained_model = AutoModelForSequenceClassification.from_pretrained(
        "./model",
        num_labels=len(CLASSES),
        id2label=ID2CLASS,
        label2id=CLASS2ID,
        problem_type = "multi_label_classification"
    )
    
    # Inference over non labeled dataset
    device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")
    trained_model.to(device)
    trained_model.eval()
    
    out_tensor = None
    n = len(TEST_DATASET["train"])
    
    for idx, entry in enumerate(TEST_DATASET["train"]):
        inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in entry.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            outputs: torch.Tensor = trained_model(**inputs)
            labels = (outputs.sigmoid()>.5).int()
            
            if out_tensor is None:
                m = len(labels)
                out_tensor = torch.empty((n, m))
            out_tensor[idx, :] = labels
    
    # Write output to csv    
    pd.DataFrame(out_tensor.numpy(), columns=CLASSES).to_csv("dataset/out.csv")
                
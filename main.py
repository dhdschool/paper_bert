from training import trainer
from data import CURRENT_DIR, CLASSES, ID2CLASS, CLASS2ID, TEST_DATASET
from transformers import AutoModelForSequenceClassification
import os
import torch
import pandas as pd
import numpy as np 

os.chdir(CURRENT_DIR)

if __name__ == "__main__":
    # Train model
    if not os.path.exists("./model/") or os.listdir("./model/") == []:
        trainer.train()
    
    # Find the most recent trained checkpoint
    checkpoints = os.listdir("./model/")
    checkpoints = [checkpoints.copy(), checkpoints.copy()]
    checkpoints[1] = list(map(lambda x: int(x.strip("checkpoint-")), checkpoints[1]))
    checkpoints = zip(*checkpoints)
    checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
    most_recent_checkpoint = checkpoints[0][0]
        
    # Load trained model
    trained_model = AutoModelForSequenceClassification.from_pretrained(
        f"./model/{most_recent_checkpoint}",
        num_labels=len(CLASSES),
        id2label=ID2CLASS,
        label2id=CLASS2ID,
        problem_type = "multi_label_classification"
    )
    
    # Inference over non labeled dataset 
    n = len(TEST_DATASET["train"])
    m = len(CLASSES) + 1
    out_tensor = torch.empty((n, m))
    out_tensor[:, 0] = torch.tensor(TEST_DATASET["train"]["ID"], dtype=torch.int64)
    
    for idx, entry in enumerate(TEST_DATASET["train"]):
        inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in entry.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            outputs: torch.Tensor = trained_model(**inputs)
            labels = ((outputs.logits.sigmoid())>.5).int()
            out_tensor[idx, 1:] = labels
        if idx % 10 == 0:
            print(f"Inference {idx}/{n}")
            
    
    # Write output to csv    
    df = pd.DataFrame(out_tensor[:, 1:].numpy(), columns=CLASSES, dtype=np.int64, index=out_tensor[:, 0].numpy().astype(int))
    df.index.name = "ID"
    df.to_csv("dataset/out.csv")
                
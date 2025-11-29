from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding
from data import CLASSES, TOKENIZER, TRAIN_DATASET, CLASS2ID, ID2CLASS
import evaluate
import os
import numpy as np 

# Metrics to compute over the eval dataset
metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
   return 1 / (1 + np.exp(-x))

# Function call provided during training
def compute_metrics(eval_pred):
   """Compute the metrics over a single evaluation example

   Args:
       eval_pred: A single dataset example

   Returns:
       dict: A dictionary containing the metrics specified by the evaluate library
   """
   x, y = eval_pred
   x = (sigmoid(x) > 0.5).astype(int).reshape(-1)
   return metrics.compute(predictions=x, references=y.astype(int).reshape(-1))

dataset = TRAIN_DATASET
data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
model_path = "microsoft/deberta-base"

# Initialize the untrained deBERTa model
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, 
    num_labels=len(CLASSES),
    id2label=ID2CLASS,
    label2id=CLASS2ID,
    problem_type = "multi_label_classification"
)


# Training hyperparameters
training_args = TrainingArguments(
   output_dir="./model",
   learning_rate=2e-5,
   per_device_train_batch_size=2,
   per_device_eval_batch_size=2,
   num_train_epochs=2,
   weight_decay=0.01,
   eval_strategy="epoch",
   save_strategy="epoch",
   load_best_model_at_end=True
)

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=dataset["train"],
   eval_dataset=dataset["test"],
   processing_class=TOKENIZER,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)


import torch
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding
from data import CLASSES, TOKENIZER, TRAIN_DATASET, CLASS2ID, ID2CLASS


dataset = TRAIN_DATASET
data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
model_path = "microsoft/deberta-base"

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, 
    num_labels=len(CLASSES),
    id2label=ID2CLASS, label2id=CLASS2ID,
    problem_type = "multi_label_classification"
)

training_args = TrainingArguments(
   output_dir="./model",
   learning_rate=2e-5,
   per_device_train_batch_size=3,
   per_device_eval_batch_size=3,
   num_train_epochs=2,
   weight_decay=0.01,
   eval_strategy="epoch",
   save_strategy="epoch",
   load_best_model_at_end=True,
)

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=dataset["train"],
   eval_dataset=dataset["test"],
   processing_class=TOKENIZER,
   data_collator=data_collator,
)

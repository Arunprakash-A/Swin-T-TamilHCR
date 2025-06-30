import argparse    
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate  
from preprocess import prepare_data, get_image_processor, preprocess_train, preprocess_val
import numpy as np
import torch

#prepare the dataset for training
path = "data/train" 
dataset, label2id, id2label = prepare_data(path)

#split training data into train and validation set
splits = dataset["train"].train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

# transform the data using torchvision
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

#load the model

def get_model(model_path):  
    
    model = AutoModelForImageClassification.from_pretrained(
        model_path, 
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True, 
    )
    return model

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

    

if __name__ == "__main__":
    
    MODEL_CHOICES = ["efficientnet","resnet-18", "resnet-50", "resnet-152","swin-t","vit"]     
    parser = argparse.ArgumentParser(description="Training configuration.")      
    parser.add_argument(
        "--model", 
        choices=MODEL_CHOICES,  
        required=True,          
        help="Choose a model configuration:efficientnet, resnet-18, resnet-50, resnet-152,swin-t or vit"
    )
    parser.add_argument(
        "--batch_size", 
        default=16,
        type=int
        help="Set the batch size based on available GPU memory"
    )

    parser.add_argument(
        "--num_train_epochs", 
        default=1,
        type=int
        help="Set the number of training epochs"
    )
    # Parse Arguments
    args = parser.parse_args()
    assert args.model in ["efficientnet","resnet-18", "resnet-50", "resnet-152","swin-t","vit"]
    model_name_to_path = {"efficientnet":"google/efficientnet-b0",
                         "resnet-18": "microsoft/resnet-18",
                         "resnet-50": "microsoft/resnet-50",
                         "resnet-152": "microsoft/resnet-152",
                         "swin-t": "microsoft/swin-tiny-patch4-window7-224",
                         "vit": "google/vit-base-patch16-224"
                         }
    model_path = model_name_to_path[args.model]
    image_processor = get_image_processor(model_path)
    model = get_model(model_path)
    
    # adjust gradient_accumulation_steps based on available compute resource
    training_args = TrainingArguments(
                                        f"{model_path}-finetuned-thcr",
                                        remove_unused_columns=False,
                                        evaluation_strategy = "epoch",
                                        save_strategy = "epoch",
                                        learning_rate=5e-5,
                                        per_device_train_batch_size=args.batch_size,
                                        gradient_accumulation_steps=4,
                                        per_device_eval_batch_size=args.batch_size,
                                        num_train_epochs=args.num_train_epochs, 
                                        warmup_ratio=0.1,
                                        logging_steps=10,
                                        load_best_model_at_end=True,
                                        metric_for_best_model="accuracy",
                                        )
    trainer = Trainer(
                        model,
                        training_args,
                        train_dataset=train_ds,
                        eval_dataset=val_ds,
                        tokenizer=image_processor,
                        compute_metrics=compute_metrics,
                        data_collator=collate_fn,
                    )
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    
    

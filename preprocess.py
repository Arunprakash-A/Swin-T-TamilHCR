import argparse
import os
import math
from PIL import Image, ImageFile
from datasets import load_dataset, Image, load_from_disk
from transformers import AutoImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


def prepare_data(path_to_data_folder):
    dataset = load_from_disk(path_to_data_folder)
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    return dataset, label2id, id2label

# pass the model_path to preprocess_xxx function. It is a higher order function
# just a trick
selected_model = None 

def get_image_processor(model_path):    
    image_processor  = AutoImageProcessor.from_pretrained(model_path)
    global selected_model 
    selected_model = model_path
    return image_processor

def normalize_(model_path):
    image_processor = get_image_processor(model_path)
    normalized = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")
    return (normalized,size,crop_size,max_size)

def preprocess_train(example_batch):
    # compose a sequence of transformations applied on input while training
    normalize,size,crop_size,max_size = normalize_(selected_model)
    train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    # Apply train_transforms across a batch.
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch
    
def preprocess_val(example_batch):
    # compose a sequence of transformations applied on validation input 
    normalize,size,crop_size,max_size = normalize_(selected_model)
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )
    #Apply val_transforms across a batch
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch
        
    
if __name__ == "__main__":
    
    MODEL_CHOICES = ["efficientnet","resnet-18", "resnet-50", "resnet-152","swin-t","vit"]     
    parser = argparse.ArgumentParser(description="Select a model configuration.")      
    parser.add_argument(
        "--model", 
        choices=MODEL_CHOICES,  
        required=True,          # Make it mandatory
        help="Choose a model configuration:efficientnet, resnet-18, resnet-50, resnet-152,swin-t or vit"
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
    
    # Print the selected model
    print(f"Selected model: {args.model}")
    print(get_image_processor(model_path))
    print(normalize_(model_path))




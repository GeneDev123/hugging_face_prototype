from datasets import load_dataset
import matplotlib.pyplot as plt
# from datasets import DatasetDict

from transformers import AutoFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer, DefaultDataCollator
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch

from sklearn.metrics import accuracy_score

from hugging_face_prototype.utils.helpers import output_seperators

def output_dataset(dataset):
  # Note: you can implement your own datasets.
  # custom_dataset = load_dataset("imagefolder", data_dir="path_to_your_dataset")

  # Dataset Loaded
  output_seperators("Dataset loaded:", dataset)

def output_sample_img(dataset):
  # Access first Image and Label from training set
  sample = dataset['train'][0]
  sample_output = {
    "image": sample['img'],
    "label": sample['label'],
  }
  output_seperators("Sample Image and Label:", sample_output)

def display_img(data):
  # Display the Image
  plt.imshow(data["image"])
  plt.title(f"Label: {data['label']}")
  plt.show()

def get_transform():
  transform = Compose([
    Resize((224, 224)),  # Resize images to match the pre-trained model input size
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image data
  ])

  return transform

# Apply transformations to dataset
def preprocess(examples):
  transform = get_transform()
  examples['pixel_values'] = [transform(image) for image in examples['img']]
  return examples

def get_pretrained_model(dataset):
  # Load a pre-trained Vision Transformer model and feature extractor
  model_name = "google/vit-base-patch16-224-in21k"
  feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
  model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=10,  # CIFAR-10 has 10 classes
    id2label={i: label for i, label in enumerate(dataset['train'].features['label'].names)},
    label2id={label: i for i, label in enumerate(dataset['train'].features['label'].names)}
  )

  return model_name, feature_extractor, model

def get_train_args():
  training_args = TrainingArguments(
    output_dir="./cifar10_classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    push_to_hub=False  # Set to True if you want to upload the model to Hugging Face Hub
  )

  return training_args

def compute_metrics(pred):
  logits, labels = pred
  predictions = torch.argmax(logits, axis=-1)
  acc = accuracy_score(labels, predictions)
  return {"accuracy": acc}
          
def initializeImageClassifier():
  print("Initializing Image Classifier")

  # Note: This code will download the dataset from hugging face and save it locally
  cifar10_dataset = load_dataset("cifar10")

  output_dataset(cifar10_dataset)
  output_sample_img(cifar10_dataset)
  display_img({"image": cifar10_dataset['train'][0]['img'], "label": cifar10_dataset['train'][0]['label']})

  # Apply preprocessing to both training and test datasets
  cifar10_dataset = cifar10_dataset.map(preprocess, batched=True,  batch_size=1000)
  
  model_name, feature_extractor, model = get_pretrained_model(cifar10_dataset)
  training_args = get_train_args()
  data_collator = DefaultDataCollator()

  trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = cifar10_dataset['train'],
    eval_dataset = cifar10_dataset['test'],
    tokenizer = feature_extractor,  # This is used to preprocess pixel values
    data_collator = data_collator,
    compute_metrics = compute_metrics
  ) 

  # Train the model
  trainer.train()

  # Save the fine-tuned model
  trainer.save_model("./cifar10_classifier")

  # Evaluate on the test dataset
  metrics = trainer.evaluate()
  print("Evaluation metrics:", metrics)



# %%
from huggingface_hub import notebook_login

# %%
#/========== PATHS and VARIABLES ==========
model_id = "google/vit-base-patch16-224-in21k"
dataset_folder = "EuroSAT/2750"


# %%
#/========== DATASET ==========
#import createDataset
import datasets
import os
import PIL
#
#dataset = createDataset.create_image_folder_dataset(dataset_folder)
#img_class_labels = dataset.features["label"].names

def create_image_folder_dataset(root_path):
    ## list directories
    _CLASS_NAMES = os.listdir(root_path)
    ## enumerate features
    features=datasets.Features({
        "img": datasets.Image(),
        "label": datasets.features.ClassLabel(names=_CLASS_NAMES),
    })
    ## create/populate list of datapoints
    img_data_files=[]
    label_data_files=[]
    for img_class in os.listdir(root_path):
        for img in os.listdir(os.path.join(root_path, img_class)):
            path_ = os.path.join(root_path, img_class, img)
            img_data_files.append(path_)
            label_data_files.append(img_class)
    ## create dataset
    return datasets.Dataset.from_dict({"img":img_data_files, "label":label_data_files}, features=features)

## Make eurosat dataset
eurosat_ds = create_image_folder_dataset(dataset_folder)
img_class_labels = eurosat_ds.features["label"].names

# %%
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#/========== PREPROCESSING ==========
from transformers import ViTImageProcessor
from tensorflow import keras
from keras import layers

#feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)
image_processor = ViTImageProcessor.from_pretrained(model_id)

data_augmentation = tf.keras.Sequential(
    [
        layers.Resizing(image_processor.size, image_processor.size),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Use keras to augment the image data: apply random transforms to the data to allow the model to generalise better.
def augmentation(examples):
    examples["pixel_values"] = [data_augmentation(image) for image in examples["img"]]
    return examples

# Basic image processing
def process(examples):
    examples.update(image_processor(examples['img'], ))
    return examples

## Process the dataset
processed_dataset = eurosat_ds.map(process, batched=True)
processed_dataset

## create test dataset 
# test size will be 30% of train dataset
test_size=.3
processed_dataset = processed_dataset.shuffle().train_test_split(test_size=test_size)

# %%
#/========== FINETUNING ==========
from huggingface_hub import HfFolder
import tensorflow as tf

id2label = {str(i): label for i, label in enumerate(img_class_labels)}
label2id = {v: k for k, v in id2label.items()}

# set hyperparameters
num_train_epochs = 5
train_batch_size = 32
eval_batch_size = 32
learning_rate = 3e-5
weight_decay_rate = 0.01
num_warmup_steps = 0
output_dir=model_id.split("/")[1]
hub_token = HfFolder.get_token()
hub_model_id = f'{model_id.split("/")[1]}-euroSat'
fp16=True
# when training on floating-point 16 set GPU policy to accomodate floating points
if fp16:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


# %%
#/=========== Convert dataset to Tensorflow dataset ===========
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")

# convert train dataset to tf.data.Dataset
tf_train_dataset = processed_dataset["train"].to_tf_dataset(
    columns=["pixel_values"],
    label_cols=["labels"],
    shuffle=True,
    batch_size=train_batch_size,
    collate_fn=data_collator
)

# convert test dataset to tf.data.Dataset
tf_eval_dataset = processed_dataset["test"].to_tf_dataset(
    columns=["pixel_values"],
    label_cols=["labels"],
    shuffle=True,
    batch_size=train_batch_size,
    collate_fn=data_collator
)

# %%
#/========== Prepare the MODEL ==========
from transformers import TFViTForImageClassification, create_optimizer
from datasets import load_metric
import tensorflow as tf
 
# create optimizer wight weigh decay
num_train_steps = len(tf_train_dataset) * num_train_epochs
optimizer, lr_schedule = create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=num_warmup_steps,
)
 
# load pre-trained ViT model
model = TFViTForImageClassification.from_pretrained(
    model_id,
    num_labels=len(img_class_labels),
    id2label=id2label,
    label2id=label2id,
)
 
# define loss
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
 
# define metrics
#metrics=[
#    keras.metrics.SparseCategoricalAccuracy(),
#    keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top-3-accuracy"),
#]

metric = load_metric("accuracy")

 
# compile model
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'],
              )

# %%
#/========== FIT MODEL ==========
train_results = model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    #callbacks=callbacks,
    epochs=num_train_epochs,
)

# %%
#/========== SAVE MODEL ==========
model.save("euroSAT.keras")

# %%
#/========== EVALUATE MODEL ==========
model.evaluate(tf_eval_dataset)

# %%
#/========== Predict label ==========
#probs = model.predict(tf_eval_dataset)
#classes = keras.np_utils.probas_to_classes(probs)


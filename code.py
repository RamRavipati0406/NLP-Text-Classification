# Import necessary libraries
from transformers import TFAutoModel
from transformers import AutoTokenizer
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load dataset
dataset = load_dataset("yelp_review_full")
dataset = dataset.shuffle(seed=0)  # Shuffle dataset
train_dataset = dataset["train"][:1000]  # Select a subset of training data
test_dataset = dataset["test"][:1000]   # Select a subset of testing data

# Load pre-trained BERT model
bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
bert_model.trainable = False  # Freeze BERT model's layers

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Set maximum sequence length for tokenization
maxlen = 512

# Tokenize and preprocess training data
tokenized_train = tokenizer(train_dataset["text"], max_length=maxlen,
                            truncation=True, padding=True, return_tensors="tf")

# Convert training labels to categorical format
train_y = to_categorical(train_dataset["label"][:1000])

# Define input layers for token IDs and attention masks
token_ids = Input(shape=(maxlen,), dtype=tf.int32, name="token_ids")
attention_masks = Input(shape=(maxlen,), dtype=tf.int32, name="attention_masks")

# Pass input through BERT model
bert_output = bert_model(token_ids, attention_mask=attention_masks)

# Define dense layer and output layer
dense_layer = Dense(64, activation="relu")(bert_output[0][:, 0])  # Extract the [CLS] token output
output = Dense(5, activation="softmax")(dense_layer)  # Softmax activation for multi-class classification

# Define the model with inputs and outputs
model = Model(inputs=[token_ids, attention_masks], outputs=output)

# Compile the model with optimizer, loss function, and evaluation metrics
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit([tokenized_train["input_ids"], tokenized_train["attention_mask"]],
          train_y, batch_size=25, epochs=3)

# Tokenize and preprocess test data
tokenized_test = tokenizer(test_dataset["text"], max_length=maxlen,
                           truncation=True, padding=True, return_tensors="tf")

# Convert test labels to categorical format
test_y = to_categorical(test_dataset["label"][:1000])

# Evaluate the model on test data
score = model.evaluate([tokenized_test["input_ids"], tokenized_test["attention_mask"]],
                       test_y, verbose=0)

# Print accuracy on test data
print("Accuracy on test data:", score[1])

# English to Hinglish Transformer

This project aims to transform English text to Hinglish (a blend of Hindi and English) using a pre-trained H5 Transformer model. It utilizes the findnitai/english-to-hinglish dataset and employs a Seq2Seq model architecture.

## Table of Contents

- [English to Hinglish Transformer](#english-to-hinglish-transformer)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Project Structure](#project-structure)
  - [Inference](#inference)
  - [Model Results](#model-results)
  - [Improvements](#improvements)

## Introduction

This is a project that uses a deep learning model to convert English sentences into Hinglish. Hinglish is a popular way of writing Hindi using the Roman script with a mix of English words.

## Requirements

To use this project, you'll need the following:

- Python 3.x
- TensorFlow 
- The FindNITAI English-to-Hinglish dataset
- Pre-trained H5 Transformer model

You can install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## Project Structure
```bash
/
├── h5_model.py             # Model Training
├── inference_testing.py    # Source code
├── hinglish_upload_v1.json # translation data foe Seq2Seq model
├── inputs.txt              # Input text file containing english sentences
├── results.json            # json file containing the results of the inference test
├── README.md               # Project documentation
├── requirements.txt        # List of required Python packages
└── LICENSE                 # License information
```

## Inference
To perform inference using the pre-trained model, follow these steps:


Prepare Input Text: Create a text file (inputs.txt) containing the English text you want to translate to Hinglish. Each line in the file should represent one input.

Run Inference Script:

```
python inference_testing.py 
```
## Model Results
Loss - 1.8073 for 5 Epochs.

## Improvements

The model can be imporved model by tuning and increasing number of epochs.
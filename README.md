# ECE 885 Artificial Neural Network Final Project

This repo are the source code and documents for the ECE 885 Artificial Neural Network final project.

## Simplified Minimal Gated Unit Recurrent Neural Network Exploration

Implement 2 variants of the MGU RNN. Compare their performance with standard LSTM, standard MGU using MNIST, NIST, and IMDB database.

The standard MGU paper is presented in "Zhou et al. Minimal Gated Unit for Recurrent Neural Networks.pdf".

## Requirement

See "ECE-CSE885_Final_Project_SS2017V.pdf"

## Method and Results

Check the report and presentation slides in "docs" folder.

## About the Source Code

### For MNIST and IMDB dataset:

Run the code directly, keras will automatically download the data from Amazon server.

### For NIST dataset:

The dataset can be downloaded from: https://s3.amazonaws.com/nist-srd/SD19/by_class.zip

### Run the Code

For the first time running the NIST program, you need to convert a subset of the raw PNG images to the `pkl` file. 

The `pkl` file is essential to the afterward network training.

To achieve this, first set the "function_flag" in "nist_main.py" to 0, and run the program, then change it to 1, and run again.

Different model name are listed bellow:

- Standard LSTM: 'lstm'
- Standard MGU: 'basic'
- MGU Variant 2: 'variant'
- MGU Variant 4: 'variant4'

The total time consumption will be printed to console at the end of the program.




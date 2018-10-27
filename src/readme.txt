# For MNIST and IMDB dataset:

Run the code directly, keras will automatically download the data from Amazon server.

# For NIST dataset:

The dataset can be downloaded from: https://s3.amazonaws.com/nist-srd/SD19/by_class.zip

For the first time running the NIST program, you need to convert a subset of the raw PNG images to the pkl file. 

The pkl file is essential to the afterward network training.

To achieve this, first set the "function_flag" in "nist_main.py" to 0, and run the program, then change it to 1, and run again.

Different model name are listed bellow:

- Standard LSTM: 'lstm'
- Standard MGU: 'basic'
- MGU Variant 2: 'variant'
- MGU Variant 4: 'variant4'

The total time consumption will be printed to console at the end of the program.




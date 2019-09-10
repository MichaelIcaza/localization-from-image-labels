# renal-carcinoma
A VGG style convolutional neural network, distinguishing chromophobic renal cell carcinoma from oncocytoma. 
The CNN uses patch-based classification with a max-pooling fusion method to classify images. 
We apply dropout onto the patch-level classifications and measure the dropout rate effect on false positive and false negative rates. 

The data used for training the model is not currently availble. During training 114 samples were used.

main.py
  contains the main experiment, executes a single fold validation
model.py
  contains the tensorflow.keras model definition
model_functions.py
  helper functions for the models, generators, callbacks, and similar
preprocess.py
  translates .stk files and stacks them into .npy arrays for a simpler training process

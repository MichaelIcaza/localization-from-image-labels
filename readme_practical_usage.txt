If you are trying to get this to run on your own files, here's some help. 

I don't suggest simply resuing most of this code, as much of it is written explicitely for the specific input dimensions, as well as the format of having 
several patients each with their own image, and distributing the patient level class approprately. 

I would suggest applying the main nuance of this paper on your own. 

In short, we apply sigmoid, dropout, then max pooling to do binary classificaiton. 

After training on the image level classes normally, we then rebuild the model to output the sigmoid activation map. These are then transformed into the original image dimensions.

In our context, we had an initial size of 400x400 (with depth, but that doesn't matter here). We designed a receptive field of size 160x160, with a stride of 40. 
(this is kind of a complicated process, there is an excel sheet here for help: https://github.com/MichaelIcaza/Excel-receptive-field-and-stride-calculations)
This causes our output to have shape 17x17. In "MODEL FUNCTIONS.PY" is a callback that shows how the sigmoid activations are mapped onto a 400x400 input shape. 

One important note is that normal dropout will scale the output by 1/(1-dropout_rate), we do not want this feature (as our probabilities would scale above 1.0)
so we use a custom lambda layer multiplying by (1-dropout_rate) to remove the scaling. 

We find that at high dropout rate, we get higher sensitivity at the cost of precision. 

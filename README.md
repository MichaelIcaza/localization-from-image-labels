# CNN Localization from Image Level Labels
A Convolutional Neural Network typically requires annotated training data to preform segmentation. In the medical field, annotating images acquired though novel means can be expensive, inconconsistent, and in some cases no experts exist to annotate our data. This project shows off a method for generating coarse grain localization using a patch-based classifier with max-pooling fusion. 

Le Hao, etall write in Patch-based Convolutional Neural Network for Whole Slide Tissue Image Classification (https://arxiv.org/abs/1504.07947) that classifying patches independently can make classification on extremely large images easier, this also allows the network to classify images too large to fit into memory with out having to downsample. Le Hao et.all explicitely suggest avoiding max pooling fusion method but we chose to use this fusion method in combination with patch-dropout. 

Our data set in this example is a set of 110 images split into two classes. The data contains tissue images acquired through MultiPhoton Microscopy, which generates 3D images with various channel frequencies. We obtain images of a benign and malignant form of renal cancer, classifying the malignant form as positive, the benign form as negative. 

Knowing only the classification of the patient, we manage to localize the signals which indicate malignancy. We believe that this method can help researchers better learn discrimitory features when applying imagining techiques in novel applications. 



![sample_output](michaelicaza.github.com/localization-from-image-labels/sample_output.png)

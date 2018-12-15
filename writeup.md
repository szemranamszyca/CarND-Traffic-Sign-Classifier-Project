# **Traffic Sign Recognition** 

## Arkadiusz Konior - Project 2.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[samples]: ./imgs/samples.png "Samples"
[samples_grey]: ./imgs/sample_grey_aug.png "Samples after modifications"
[distro_before]: ./imgs/distro_before.png "Label's frequency histogram "
[distro_after]: ./imgs/distro_after.png "Label's frequency histogram"
[scaled]: ./imgs/scaled.png "Example of scaled image"
[rotated]: ./imgs/rotated.png "Example of rotated image"
[blured]: ./imgs/blured.png "Example of blured image"
[combined]: ./imgs/combined.png "Example all methods combined"
[softmax]: ./imgs/softmax_prop.png "Softmax probability charts"
[internet]: ./imgs/internet.png "My set of signs found on Internet"
[lenet]: ./imgs/lenet.png "LeNet architecture"
 	



---
### Writeup / README

Link to my [project code](https://github.com/szemranamszyca/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Dataset.

To extract information about dataset, I've just used Python functions. At this moment, validation set is loaded from the provided file

- Number of training examples = 34799  
- Number of validating examples = 4410  
- Number of testing examples = 12630  
- Image data shape = (32, 32, 3)  
- Number of classes = 43  

#### 2. Visualization of the dataset.

Before any operation, there are few examples of signs from dataset:
![Samples][samples]

Histogram of samples at the beginning:
![Histogram][distro_before]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

As my first step, I've converted all images to greyscale, as it was suggested at *Traffic Sign Recognition with Multi-Scale Convolutional Networks*

To augment data, I've written some functions to help me with that:
+ blur  
![Blured image][blured]
+ rotate  
![Rotated image][rotated]
+ scaling   
![Scaled image][scaled]

All of them can be found in 5-8 cells.

As the last step for augmentation, I combined all of these methods on one image, here's the example:
![Combined all of methods][combined]

I normalized data using suggested method. During augmentation process I want to reduce disproportion between classes. As you can see at histogram, there's a huge difference in amout of examples for the first and the second sign. During augmentation process I set minimal amount of each class to 1200 examples.

![New histogram][distro_after]

Also, I've decied to get new validation set, split from augmented examples. After all of these operations, dataset parameters are:

- Number of training examples = 45888  
- Number of validating examples = 11472  
- Number of testing examples = 12630  
- Image data shape = (32, 32, 1)  
- Number of classes = 43 

Here's the set of example images after all these modficiations:

![Samples after augmentation][samples_grey]


#### 2.Final Model architecture

I've decided to use classical LeNet architecture and just tune it a little, adding the dropout method (with probablity 0.75) after Layer 3 and Layer 4


| Layer         		      |     Description	        					                                    | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 RGB image   							| 
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6 	| 
| RELU					             |												| 
| Max pooling	      	   | 2x2 kernel size, 2x2 stride,  input = 28x28x6, outputs 14x14x6    | 
| Convolution 5x5	      | 1x1 stride, valid padding, input = 14x14x16, outputs 10x10x16  		| 
| RELU					             |												|
| Max pooling	      	   | 2x2 kernel size, 2x2 stride,  input = 10x10x16, outputs 5x5x16    | 
| Flatten               | input = 5x5x16, output = 400 | 
| Fully connected		     | Input = 400. Output = 120 |
| RELU					             |												| 
| Dropout		             |	keep probability = 0.75											| 
| Fully connected		     | Input = 120. Output = 84 |
| RELU					             |												| 
| Dropout		             |	keep probability = 0.75											| 
| Fully connected		     | Input = 84. Output = 43 | 
| Softmax				           |         									| 
 
![LeNet Architecture][lenet]

#### 3. Model train parameters.


As I mentioned before, I've already used implemented LeNet architecture with Adam optimizer. These are my settings:
- Batch size: 100
- Epochs: 200
- Learning rate: 0.0001
- Mu: 0
- Sigma: 0.1
- Dropout keep probability: 0.75

#### 4. Disscusion
Authors of *Traffic Sign Recognition with Multi-Scale Convolutional Networks* wrote that LeNet neural network architecture could be used as traffic sign classificator. However, even after converting it to greyscale and augmented data, validation accuracy was still not enough (a little more than 90%).  
I decided to add a dropout layer with "keep probability 0.75" and that gives me a result of 93.1% for validation set and 88.1% for test set.
 

### Test a Model on New Images

#### 1. Signs found on the Internet

Here are ten German traffic signs that I found on the web:

![My signs][internet]

All signs were resized to 32x32 resolution. This could make details little fuzzy - like *Road work* or *Right-of-way at the next intersection*. Few of them have quite low contrast (*Priority road*, *Yield*, *No entry*). The image presents the *No entry* sign from an angle perspective, and it might be a problem. Also, digit 5 from *Speed limit (50km/h)* could be easily confused with 3.


#### 2. Discussion

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		                       | No entry   				     	|   
| Stop      			                          | Stop 						               				|   
| Right-of-way at the next intersection  | Beware of ice/snow, probability 											|  
| Bumpy road	                            | Bumpy Road					 			 	|  
| Yield			                               | Yield     					    		|  
| Speed limit (70km/h)                   | Keep left   									|   
| Road work                              | Road work   									|   
| No vehicles                            | No vehicles 										|   
| Priority road                       			| Priority road											|  
| Speed limit (50km/h)	                  | Speed limit (30km/h)						|  

The model guessed correctly 7 of the 10 traffic signs, which gives us an accouracy of 70%. This is less than test set accuracy. This result is quite poor, and if I ever want to develop my own autonomous vehicle, I should improve it. At this moment I could be really dangerous on the road for other cars.

#### 3. The top 3 softmax probabilities for each image.

![Softmax probabilities][softmax]

As I've written in previous paragraph, 7 of 10 signs were guessed correctly with almost always 100% probability. Let's discuss  an interesting part: why 3 of them where predicted wrongly.  

*Right-of-way at the next intersection* was confused with *Beware of ice/snow*, but still  - nerual network gives a 10% for correct guess. With such poor quality of images, indeed, NN could see some similarities of these two images. Considering the fact that accuracy of the test set is not so high (88%), I get this error.  

Very similar situation is with *speed limit 50km/h*, which was taken by mistake as 30km/h. The same situation is here: image quality makes 50 and 30 very similar to each other.

However, the biggest surprise is the last error: *Speed limit (70km/h)* was classified as *Keep left* sign, with probability 96.052%! The first thing that came to my head was that something might be wrong with the train data - maybe data during augmentation was shifted in dataset? To improve my result, I should investigate carefully this error.

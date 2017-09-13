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
[internet]: ./imgs/softmax_prop.png "My set of signs found on Internet"
[lenet]: ./imgs/lenet.png "LeNet architecture"
 	



---
### Writeup / README

Link to my [project code](https://github.com/szemranamszyca/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Dataset.

To extract information about dataset, I've just used python functions. At this moment, validation set is loaded from provided file

- Number of training examples = 34799  
- Number of validating examples = 4410  
- Number of testing examples = 12630  
- Image data shape = (32, 32, 3)  
- Number of classes = 43  

#### 2. Visualization of the dataset.

Before any operation, here's few examples of signs from dataset:
![Samples][samples]

Histogram of samples at the beginning:
![Histogram][distro_before]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

As a first step, I've converted all images to greyscale, as it was suggested at *Traffic Sign Recognition with Multi-Scale Convolutional Networks*

To augmented data, I've written some function to help mi with that:
+ blur  
![Blured image][blured]
+ rotate  
![Rotated image][rotated]
+ scaling   
![Scaled image][scaled]

All of them you can find in 5-8 cells.

As a last step for augmentation I combined all of these method on one image, here's the example:
![Combined all of methods][combined]

I normalized data using suggested method. During augmentation process I want to reduce disproportion between classes. As you can see at histogram, there's a huge difference in amout of examples for first and second sign. During augmentation process I set minimal amount of each class to 1200 examples.

![New histogram][distro_after]

Also, I've decied to get new validation set, splitted from augmented examples. After all of these operations, dataset parameters are:

- Number of training examples = 45888  
- Number of validating examples = 11472  
- Number of testing examples = 12630  
- Image data shape = (32, 32, 1)  
- Number of classes = 43 

Here's a set of example images after all of these modficiations:

![Samples after augmentation][samples_grey]


#### 2.Final Model architecture

I've decided to use classical LeNet architecture and just tune it a little, adding to dropout method (with probablity 0.75) after Layer 3 and Layer 4


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


As I mentioned before, I used already implemented LeNet architecture with Adam optimizer and here's my settings:
- Batch size: 100
- Epochs: 200
- Learning rate: 0.0001
- Mu: 0
- Sigma: 0.1
- Dropout keep probability: 0.75

#### 4. Disscusion
In *Traffic Sign Recognition with Multi-Scale Convolutional Networks* paper authors wrote, that LeNet neural network architecture could be used as traffic sign classificator. However, even after converting it to greyscale and augmented data, validation accuracy was still not enough (a little more then 90%).  
I decided to add to dropout layers with keep probability 0.75 and that gives me a result of 93.1% for validation set and 88.1% for test set.
 

### Test a Model on New Images

#### 1. Signs find on Internet

Here are ten German traffic signs that I found on the web:

![LeNet Architecture][internet]

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

The model guessed correctly 7 of the 10 traffic signs, which gives an accouracy of 70%. This is less than test set accuracy. This result is quite poor and if I ever want to developed my own autonomous vehicle I should improve it because for this moment I could be a really dangerous on the road for other cars.

#### 3. The top 3 softmax probabilities for each image.

![Softmax probabilities][softmax]

As I've written in previous paragraph, 7 of 10 signs were guessed correctly with almost always 100% probability. Let's discuss about interesting part - why 3 of them where predicted wrong.  

*Right-of-way at the next intersection* was confused with *Beware of ice/snow*, but still  - nerual network gives a 10% for correct guess. At this poor quality of images, indeed, NN could see some similarities of these two images. Adding the fact, that accuracy of test set is not so high (88%), I get this error.  

Very similar situation is with *speed limit 50km/h*, which was by mistake, taken as 30km/h. The same - image quality makes 50 and 30 very similar to each other.

However, the biggest surprise is last error - *Speed limit (70km/h)* was classified as *Keep left* sign, with probability 96.052%! The first thing that came to my head, that something may be wrong with train data - maybe data during augmentation was shifted in dataset? To improve my result and should investigate carefully this error.

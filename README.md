# Deep-Learning-Workshop
The repository contains python codes, which I have developed as the facilitator of the two consecutive Deep Learning Workshops (I and II) for the master's students of computer science, University of Windsor.

Workshop on Deep Learning
facilitator 
Shaon Bhatta Shuvo
PhD Student, School of Computer Science
University of Windsor, Ontario, Canada.
Email: shuvos@uwindsor.ca
----------------------------------------------------------------------------------------------------
Objective:
With the availability of data and the increase of computational power, the applications of deep learning and the popularity is increasing exponentially. The main objective of this workshop is to provide introductory theoretical knowledge on deep learning and practical implementations using python programming language and associated packages. Therefore, participants will get an overall idea of deep learning and its applications. This workshop will also guide students on what they need to learn to able to build their careers in this field. Overall the workshop shall work as a useful starting point and will encourage participants to dig deeper into the field of deep learning.
Intended Learning Outcomes:
At the end of this workshop, active participants should be able to: 
1.	Identify the importance and applications of deep learning. 
2.	Choose between machine learning and deep learning approaches to solve a particular problem.
3.	Demonstrate basic concepts of deep learning workflow. 
4.	Build deep learning models from scratch. 
5.	Evaluate the model’s performance. 
6.	Demonstrate fundamental concepts of some state-of-the-art deep learning algorithmic approaches and implement those to different datasets. 
7.	Tune parameters to optimize the model’s performance. 
8.	Identify what to learn to become an expert in this field. 
 

Prerequisites:
1.	The participants are expected to have a minimum basic understanding of python programming language. 
2.	Basic knowledge of Machine Learning with help to understand the concept with less effort.  
Required Tools/Setup:
To follow the implantations along with the instructor and to avoid any unwanted delay, the participants are advised to install following tools/packages before joining the workshop:
1.	Strongly Recommend: 
-	Python (Latest Version)
-	Anaconda (Latest Version)  
2.	Better to have the latest version of the following packages installed to avoid unnecessary delay/interruption amid the workshop: 
-	tensorflow
-	keras
-	numpy
-	pandas
-	matplotlib
-	scikit-learn
-	mlxtend

N.B: If any participant does not have the above setup, they can also use Google Colab to run the codes. However, the cloud service may not support a few packages (e.g., mlxtend).







Workshop (Part1) 						    Duration: 3 hours 
Tentative Topics to Cover 						    Approximate Duration
1.	Why we need Deep Learning?						     10 mins.
2.	What is Deep Learning? 					            	      5 mins. 
3.	Machine Learning vs Deep Learning. 					      5 mins.
4.	Looking back to history.							     10 mins.
- God Fathers of Deep Learning
- Is Deep Learning a new concept? 
- Why Deep Learning is getting popular now?
5.	Concept Neuron, Artificial Neuron and Neural Networks			      5 mins
- Similarities between Biological Neuron and Artificial Neuron.
6.	Why do we call it deep learning? 						      5 mins
	- Concept of Layer
	- Examples of a Deep Neural Network						
7.	How Neural Networks work?						      	      10 mins
-Non-linearity and Activation Functions
8.	Ingredients to train a Deep Neural Network					      30 mins
- Data 
- Model
- Objective Functions 
- Optimization Algorithms 
- Back Propagation	
- Learning Rate
- Importance of Gradient Computation.
9.	Overview: How do Neural Networks Learn?					      10 mins
-Summarizing and putting all the ingredients together.  
10.	Parameters vs Hyperparameters						      5 mins

11.	Necessary tools and set up for deep learning implantation.			      10 mins
-   Installing python and anaconda	
-   Installing all the required packages (tensorflow, keras etc.). 
12.	Building the first ANN model from the scratch 				      30 mins
-	Synthetic Dataset Generation (Linear Data for Regression)
-	Dataset Visualization
-	Data Preprocessing
-	Build the Model 
-	Train the Model 
-	Performance Evaluation 
-	Performance Visualization 
-	Tuning Hyperparameters 
13.	Comparison of Models Performance 						      5 mins
-	Comparison with traditional ML algorithms (e.g. Linear Regression)
14.	Building the second first model from the scratch 				      20 mins
-	Synthetic Dataset Generation (Non-Linear Data for Classification)
-	Data Visualization
-	Data Preprocessing
-	Concept of separate Validation Set
-	Build the Model 
-	Train the Model 
-	Performance Evaluation 
-	Performance Visualization 
-	Tuning Hyperparameters 
15.	Comparison of Models Performance 						   10 mins
-	Comparison with traditional ML algorithms (e.g. SVM Classifier)
16.	Question Answering 								   10 mins

Workshop (Part2) 					          Duration: 3 hours 
Tentative Topics to Cover 						    Approximate Duration
1.	Overview of deep learning (based on Workshop 1)				10 mins
2.	Deep Convolution Neural Network (CNN)					30 mins
-  Why Convolutional Neural Network?
-  What is Convolutional Neural Network? 
-  Convolutional Operation
-  RelU Layer
-  Pooling
-  Flattening
-  Fully Connected Layer
-  Loss Function 
-  Output Layer’s Activation Function
-  Backpropagation
-  Summarization
-  Some popular CNN Models
3.	Handwritten Character Recognition on MNIST Dataset using CNN (Hands-on) 40 mins
		-  Downloading the dataset from Keras library
		-  Visualizing a portion of Dataset 
		-  Preprocessing the dataset	
-  Concept of Validation Dataset 
		-  Splitting the dataset into train, test and validation set. 
		-  Build the model 
		-  Model Compilation
		-  Train the model
		-  Performance Evaluation on Test set
		-  Performance Visualization 
4.	Comparing the performance with Multi-Layer NN model 			    10 mins
5.	Concept of Transfer Learning (TL) 						    10 mins
-	What is Transfer Learning?
-	Why this is useful?
6.	Canadian Medicinal Plant Recognition (Hands-on) using CNN with TL        40 mins
-	InceptV3 model for transfer learning		
-	Collecting dataset manually from the internet 
-	Preprocess the data
-	Plotting the proportion of each category (train, test, validation) using pie chart
-	Build the model 
-	Model Compilation
-	Train the model 
-	Performance Evaluation on Test set
-	Performance Visualization 
7.	Concept of Overfitting							      20 mins.
-	What is overfitting?
-	Techniques to avoid overfitting.
-	Improving previously implemented model’s performance using Regularization Techniques e.g. Dropout and Early Stopping.  		
8.	What needs to know to understand deep learning more efficiently? 	      10 mins.
9.	Question Answering 								      10 mins.

N.B: All the implementation details (codes, dataset) will be provided through GitHub repositories(https://github.com/ShaonBhattaShuvo/Deep-Learning-Workshop). Therefore, if any participant does not have the proper setup at that particular moment, they can download the codes, dataset, and implement those in their pc as per their convenience. In this case, such participants should follow the session and take proper notes.

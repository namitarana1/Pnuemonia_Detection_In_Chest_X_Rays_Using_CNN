# Pnuemonia_Detection_In_Chest_X_Rays_Using_CNN
[![depositphotos-48099843-stock-photo-pneumonia-normal-alveoli-vs-pneumonia-2.jpg](https://i.postimg.cc/rmVk6tsf/depositphotos-48099843-stock-photo-pneumonia-normal-alveoli-vs-pneumonia-2.jpg)](https://postimg.cc/d7xXr38C)


Author           : Namita Rana

### Project Name    

Detection of Pneumonia from Chest X-Ray Images using ConvolutionalNeural Network and Transfer Learning.


[![photo-1584555613582-7cbc3b14a376.jpg](https://i.postimg.cc/BZzdpzTY/photo-1584555613582-7cbc3b14a376.jpg)](https://postimg.cc/9zG8FJNZ)
                   
#### Dataset Details

Dataset Name            : Chest X-Ray Images (Pneumonia)

Number of Class         : 2

Number/Size of Images   : Total      : 5863 
                          Training   : 5216 
                          Validation : 16  

# Business Problem: 
Building a model that can classify whether a given patient has pneumonia, given a chest x-ray image.

Stakeholer: Imaging labs/ Hospitals.

Business Questions: How can a successful model help save medical professionals time, money and promote better accuracy in patient diagnosis.

## Background:


Pneumonia is an infection that inflames your lungs' air sacs (alveoli). The air sacs may fill up with fluid or pus, causing symptoms such as a cough, fever, chills and trouble breathing. Bacteria and viruses are the main causes of pneumonia. Pneumonia-causing germs can settle in the alveoli and multiply after a person breathes them in. Pneumonia can be contagious. The bacteria and viruses that cause pneumonia are usually inhaled.
Commonly affected are Infants, children and people over 65 years in age.



Chest X-rays are used for detecting the Pneumonia infection and to locate the infected area in the lungs. So, To detect the the pneumonia radiologist have to observe the chest xray and he/she has to update the doctor correctly. The main objective of this model is to identify if the person has Pneumonia or not with high accuracy so that the person can get treatment as soon as possible. Deep Learning models which are trained correctly by using good datasets can be helpful for doctors. 

To train the model for detecting whether the person has pneumonia or not, A Convolutional Neural Network(CNN) is used. The CNN can train the images of chest xrays and then it can predict with high accuracy.



## Data Structure, Selection & Transformation:

The dataset we recieved in Kaggle is actually distributed into 3 folders (train, test, val) and individually, they contain subfolders for each image category (Pneumonia/Normal).

There are a total of 5,863 X-Ray images (in JPEG Format) distributed into 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

## Methods
## Cleaning and Feature Engineering


This project uses data cleaning and feature engineering to also addressed the class imbalance between classes we have used Data Augmentation.


Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It acts as a regularizer and helps reduce overfitting when training a machine learning model.


In order to avoid overfitting problem, we need to expand artificially our dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations. Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more. By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.



### Models Development
We have implemented versions of CNN's with different parameters,dense layers,dropout layers to see how results varies with each change in the parameters.

### Metrics used:

1. Accuracy
2. Recall

Here, "Recall" is the most significant metric even more than accuracy and precision. False negative has to be minimized because falsely diagnosing a patient of pneumonia as not having pneumonia is a much larger concern than falsely diagnosing a healthy person as a pneumonia patient. By minimizing false negative, which is in the denominator, we can increase 'Recall' .This model achieves a Recall of 99%.


## Libraries

Languages               : Python

IDE               : Anaconda/Jupyter-Notebook

Libraries               : Keras, TensorFlow

#### Results

Results
Our model,CNN came back with a confusion matrix that produced a 91% accuracy score and a 99% recall score. For our purposes, we were looking to minimize recall as we want to reduce the amount of False positives (False negatives: Patients got negative results but has actually has Pneumonia).


##Model Parameters
Machine Learning Library: Keras
Base Model              : Custom Deep Convolutional Neural Network

Optimizers              : Adam,rmsp

Loss Function           : binary_crossentropy

For Custom Deep Convolutional Neural Network : 
Training Parameters
Batch Size              : 32
Number of Epochs        : 12
Training Time           : 1.3 Hours

Output (Prediction/ Recognition / Classification Metrics)
Testing
Accuracy (F-1) Score    : 91.53%
Loss                    : 0.41
Precision               : 93.0%
Recall (Pneumonia)      : 99.0% (For positive class)



## Sample Output:
[![Screen-Shot-2022-02-19-at-1-22-12-AM.png](https://i.postimg.cc/02FP6QXN/Screen-Shot-2022-02-19-at-1-22-12-AM.png)](https://postimg.cc/kBQkznmr)


## Confusion Matrix of the selected model:

[![Screen-Shot-2022-02-19-at-1-14-16-AM.png](https://i.postimg.cc/NF0BXrwK/Screen-Shot-2022-02-19-at-1-14-16-AM.png)](https://postimg.cc/hzNkWvvn)

#### Future Work
● Training selected models with a a higher no of epochs to try to reach convergence.

● Gathering more data for a better model.

● Testing this data on different models.

● This work can be extended to detect and classify X-Ray images with lung cancer & Pneumonia.

## For More Information

Please review my full analysis in [my Jupyter Notebook] (https://github.com/namitarana1/Pnuemonia_Detection_In_Chest_X_Rays_Using_CNN/blob/master/Notebook/Pneumonia_Detection_InChest_X_Rays_using_CNN.ipynb).

For any additional questions,
please contact:
- Namita Rana at <namitarana21@gmail.com>

## Repository Structure

```
├── Images<- 
├── Docs
├── Pneumonia_Detection_InChest_X_Rays_using_CNN.pptx                               
└── Pneumonia_Detection_InChest_X_Rays_using_CNN.ipynb                           
```

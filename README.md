# Facial-Expression-Recognition-using-Haar-Cascade-Classifier
Facial Group expression Recogniser along with age and gender by using one of the IBM Services Watson Assistant.

Updated Youtube link: https://youtu.be/LIKF8lMTxCE
Preview link of chatbot creation:https://web-chat.global.assistant.watson.cloud.ibm.com/preview.html?region=eu-gb&integrationID=770935a9-c985-4c7b-87fe-1a4c928751dc&serviceInstanceID=a83ed73b-189b-4336-a098-65cf38969af6

Step 1: The block diagram tells us about the working of our system. When we consider 
the first three inputs i.e., facial expression , age, gender are trained into a dataset. 

Step 2: When comes to preprocessing, the image is taken from the live video and it is 
converted to vector form. So that they should be in the binary form. In this preprocessing 
step removing of blurred images, negative images, and noise will be done.
Here we also perform data augmentation where it is used to increase the number of 
training images in the dataset.

Step 3: The next step is the feature extraction, here we only process the relevant 
information. Here we crop the image , rotate the image to extract the ideal features that 
are relevant to our dataset. 

Step 4: In the preprocessed dataset we attain a perfect image without any noise. The 
preprocessed dataset is divided into training and test to get the accurate results. 
Some datasets are considered under training to recognize the model and some are 
considered under test to predict the results.

Step 5: The outputs that we get from the training and the test are given to the CNN where 
it takes an image as input and assigns the importance to differentiate one feature from 
another. 
Here we also use Haar Cascade algorithm which helps in finding face.

Step 6: Here we used Open CV which opens window screen. Now the extracted image of 
the face goes to the already build model for prediction. 
In simple words when we run the program the open CV helps to open a webcam which 
takes the face as input and gives the relevant output using the datasets predicting the 
facial expression, age, gender respectively.

By performing all the above steps. We have also added one of the IBM services i.e., Watson Assistant which helps in knowing about our project more.

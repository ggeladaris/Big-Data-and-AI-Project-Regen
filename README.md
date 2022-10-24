# Big-Data-and-AI-Project-RegenService for Airbnb Hosts in Athens
This project was developed as the final group project on the Regeneration Academy seminar on Big Data and Artificial Inteligence (November 2021). It is a case study for creating a service for Airbnb hosts in Athens using Microsoft Azure. This aspect of the project focuses on creating and training a model that will predict the price of listings and is based on data collected from the listings already on the site.
The Code
The Python code is split in 4 different modules:

    Preprocessing: where the functions necessary for preprocessing the given data are defined
    Splitting: where the functions necessary to split the data into the desired training and testing parts for the model are defined
    Training-Evaluating: where the functions necessary to create and train the model are defined, as well as the ones that will messure its accuracy and create predictions for the test dataset. 

Inside, each of these modules exists a function (Preprocessing, Splitting and Train_Eval respectively) that uses the functions created as its name suggests, that is preprocessing and splitting the dataset, as well as training a model and applying it to the test set. These chief functions in each module are imported and used in the main function of the project whereupon the accuracy of the model is returned.

In order to test the model again one just has to call the main function of the main module. It is important that the script files are in the same directory(folder) and of course so should the data.csv file containing the listings be. 

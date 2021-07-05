# Capstone Project for Azure Machine Learning Engineer Nanodegree in Udacity

## Table of Content
* [Overview](#overview)
* [Project Set Up and Installation](#project-set-up-and-installation)
* [Dataset](#dataset)
    * [Dataset overview](#dataset-overview)
    * [Task](#task)
    * [Access](#access)
* [Automated ML](#automated-ml)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Screen Recording](#screen-recording)
* [Standout Suggestions](#standout-suggestions)

## Overview

This is the capstone project for the "Machine Learning Engineer for Microsoft Azure" Udacity's Nanodegree. 

In this project, I chose an external dataset from a [Kaggle Competition](https://www.kaggle.com/c/interbank20) organized by a Peruvian Bank where the objective was to predict the default score for the customers. I just used two of the different datasets that the competition offered to the participants and made a preprocessing to get a unique train dataset. 

This dataset will be used to train a model using an Automated ML and a Hyperdrive. After that, I compared the performance of the
two different algorithms and deploy the best model. Finally the endpoint produced was used to get some answers about predictions.

**Figure 1**: Kaggle competition
<img src="img/datathon-ibk.png" width="800">

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

### Project Files 
In this repo you will find the following files, which were required to run the experiments:
* automl.ipynb :  Notebook file for the AutoML.
* endpoint.py : This is the python script I used to consume the produced endpoint.
* train.py : A python script that the HyperDrive operates on in order to produce the runs and find the best model.
* hyperparameter_tuning.ipynb : This is the notebook file I used for the HyperDrive. 
* dataset_train.csv : Dataset that I used from [here](https://www.kaggle.com/c/interbank20/data).

The following came out from the running of the experiments:
* model.pkl :  The best model from the AutoML I downloaded from Azure ML studio.
* score.py : I downloaded this script from Azure Machine Learning Studio and it is used to deploy the model.
* hyper-model.pkl : This is the best model from the HyperDrive I downloaded from Azure ML studio.

## Dataset

### Dataset overview
The dataset I used is from a [Peruvian Kaggle Competition](https://www.kaggle.com/c/interbank20) where there are like 4 datasets. I just used two of them: rcc_train (dataset about the debts that customers has in the financial system) and se_train (socio economic features about customers). I pre-processed the data before I uploaded to the Azure Machine Learning worskpace in the dataset-preprocessing.ipynb notebook. It describes the financial behaviour of the customers. The data have almost 355K rows of these behaviours recorded from customers.

**Figure 2**: Dataset sample

<img src="img/dataset-overview.png" width="800">

### Task
I am using this data in order to predict the default event for a customer, that means whether or not the customer will pay its debt.

The features of the data are the following:

* edad: Age of the customer. This feature was normalized.
* est_cvl: Civil status of the customer.
* sit_lab: Employment status of the customer.
* ctd_hijos: Number of customer children.
* flg_sin_email: Flag that expresses whether or not it has an email address.
* ctd_veh: Number of vehicles of the customer.
* tip_lvledu: Type of educational level.
* total_mean_of_saldo_count_per_month: Average # of debts for all months.
* total_sum_of_saldo_count_per_month: Total # of debts for all months.
* total_min_of_saldo_count_per_month: Min # of debts for all months.
* total_max_of_saldo_count_per_month: Max # of debts for all months.
* total_mean_of_saldo_sum_per_month: Average of total amount debt per month for all months.
* total_sum_of_saldo_sum_per_month: Total sum of total amount debt per month for all months.
* total_min_of_saldo_sum_per_month: Min of total amount debt per month for all months.
* total_max_of_saldo_sum_per_month: Max of total amount debt per month for all months.
* total_min_of_saldo_min_per_month: Min amount debt for all months.
* total_mean_of_saldo_min_per_month: 'Minimum amount debt per month' average for all months. 
* total_max_of_saldo_max_per_month: Max amount debt for all months.
* total_mean_of_saldo_max_per_month: 'Maximum amount debt per month' average for all months. 
* total_min_of_condicion_min_per_month: days of minimum delay in all months
* total_mean_of_condicion_min_per_month: 'Minimum delay per month' average for all months.
* total_max_of_condicion_max_per_month: days of maximum delay in all months
* total_mean_of_condicion_max_per_month: 'Maximum delay per month' average for all months.
* total_mean_of_condicion_mean_per_month: 'Mean delay per month' average for all months.

### Access
I upload the dataset in the Azure ML studio from local file **dataset_train.csv** manually. As you can see in either the automl.ipynb and hyperparameter_tuning.ipynb the code is checking whether or not the .csv has been uploaded.

**Figure 3**: Registered dataset in the workspace
<img src="img/dataset-list.png" width="800">

**Figure 4**: Details about the Dataset
<img src="img/dataset-detail.png" width="800">

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

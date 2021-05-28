# Project #2: Operationalizing Machine Learning

## Table of Content
* [Overview](#overview)
* [Architectural Diagram](#architectural-diagram)
* [Key Steps](#architectural-diagram)
    * [Authentication](#authentication)
    * [Automated ML Experiment](#automated-ml-experiment)
    * [Deploy the best model](#deploy-the-best-model)
    * [Enable logging](#enable-logging)
    * [Swagger Documentation](#swagger-documentation)
    * [Consume model endpoints](#consume-model-endpoints)
    * [Create and publish a pipeline](#create-and-publish-a-pipeline)
* [Screen Recording](#screen-recording)
* [Standout Suggestions](#standout-suggestions)

## Overview
This is the second project of the Udacity Machine Learning Engineer with Microsoft Azure Nanodegree Program. I continued my work with the [Bank Marketing](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) dataset, but this time I used Azure to configure a cloud-based machine learning production model, deploy it and consume it. I also created, published, and consumed a ML pipeline in order to show how we can automate the deployment of a ML model using Python SDK.

## Architectural Diagram
These are the steps I followed in this project :

**Figure 1**: Mains steps for the Project
<img src="img/main-steps.png" width="800">

1. **Authentication** : I created a Service Principal (SP) in order to interact with the Azure ML Workspace because I worked in the free 30-day trial subscription that Azure offers when you create a new account.
2. **Automated ML Experiment** : I created an experiment using Automated ML, configure a compute cluster, and use that cluster to run the experiment.
3. **Deploy the best model** : I deployed the Best Model in order to interact with its HTTP API service, that is to interact with the model by sending data over POST requests.
4. **Enable logging** : I enabled logging because it helps me to monitor the deployed model in order to know the number of requests it gets, the time each request takes, etc.
5. **Swagger Documentation** : I consumed the deployed model using Swagger.
6. **Consume model endpoints** : I interacted with the endpoint using some test data to get inference.
7. **Create and publish a pipeline** : I automate this workflow by creating a pipeline with the Python SDK.

## Key Steps

### Authentication
I used the free 30-days subscription offered by Azure, so I had to do this step and create a Service Principal and associate it with the Azure ML Workspace.

First, I opened a cloud shell from the portal azure and install de Azure ML extension. After that, I created the Service Principal (SP).

**Figure 2**: Install the Azure ML extension and create SP
<img src="img/authentication/auth-1.png" width="800">

Then with the Object Id of the new Service Principal, I enabled its access to the workspace.

**Figure 3**: Allow the SP access to the workspace
<img src="img/authentication/auth-2.png" width="800">

### Automated ML Experiment
In this step, I created an AutoML experiment to run using the **Bank Marketing** Dataset which was loaded in the Azure Workspace, choosing **'y'** as the target column.

First, I uploaded this dataset into the Azure ML Studio in the *Registered Dataset* Section using the url provided in the project.

**Figure 4**: Uploading from the dataset URL
<img src="img/auto-ml-experiment/automl-exp-1.png" width="800">

**Figure 5**: Settings and preview of the dataset
<img src="img/auto-ml-experiment/automl-exp-2.png" width="800">

**Figure 6**: Schema of the dataset
<img src="img/auto-ml-experiment/automl-exp-3.png" width="800">

**Figure 7**: Confirm details

<img src="img/auto-ml-experiment/automl-exp-4.png" width="800">



**Figure 8**: Registered dataset


<img src="img/auto-ml-experiment/automl-exp-5.png" width="800">

After that, I created the Auto ML run choosing the recently created dataset.

**Figure 9**: Selecting dataset in the Auto ML Run
<img src="img/auto-ml-experiment/automl-exp-6.png" width="800">

For the compute cluster, I configured the size of **Standard_DS12_v2** for the Virtual Machine and 1 as the minimum number of nodes. 

**Figure 10**: Selecting the vm size for the compute cluster
<img src="img/auto-ml-experiment/automl-exp-7.png" width="800">

Due to I was working on a free subscription I coulnd't configurate more than 1 node in the *Maximum number of nodes*, but it wasn't a big problem. The experiment took approximately 1 hour.

**Figure 11**: Configuring # of nodes
<img src="img/auto-ml-experiment/automl-exp-8.png" width="800">

**Figure 12**: Configuring the experiment run
<img src="img/auto-ml-experiment/automl-exp-9.png" width="800">

I ran the experiment using classification, without enabling Deep Learning. I enabled the *explain best model* option in order to interpret the results from the best model.

**Figure 13**: Additional configurations
<img src="img/auto-ml-experiment/automl-exp-10.png" width="800">


**Figure 14**: Select task type


<img src="img/auto-ml-experiment/automl-exp-11.png" width="800">

The run took approximately 1 hour to test various models and found the best model for the task. The best algorithm found is the **votingEnsemble** with an accuracy of 92%.

**Figure 15**: Experiment run details

<img src="img/auto-ml-experiment/automl-exp-12.png" width="800">

### Deploy the best model
In order to interact with the best chosen model for our task, we need to deploy it. This can be easily done in the Azure Machine Learning Studio, which provides us with an URL to send our test data to.

In this step, I deployed the trained Voting Ensemble model using Azure Container Instance (ACI), with authentication enabled.

**Figure 16**: Best Model


<img src="img/deployment/deployment-1.png" width="800">

**Figure 17**: Deployment of the model
<img src="img/deployment/deployment-3.png" width="800">

The model was successfully deployed. So, I could access the model endpoint in the Endpoints section of Azure ML Studio.

**Figure 18**: Real-time endpoints
<img src="img/deployment/deployment-5.png" width="800">


### Enable logging
I could enabled the Application Insights and Logs at the time of the model deployment, however for this project I achieved it using Azure Python SDK through the logs.py file.

**Figure 19**: Updating logs.py


<img src="img/logging/logging-1.png" width="800">


**Figure 20**: Config.json and logs.py
<img src="img/logging/logging-2.png" width="800">

**Figure 21**: Executing logs.py


<img src="img/logging/logging-3.png" width="800">

**Figure 22**: Results from logs.py execution
<img src="img/logging/logging-4.png" width="800">

By running the logs.py script, **I enabled Application Insights**.

**Figure 23**: Application Insights enabled for the endpoint
<img src="img/logging/logging-5.png" width="800">

### Swagger Documentation
To consume our best AutoML model using Swagger, I first needed to download the **swagger.json** file provided to us in the Endpoints section of Azure Machine Learning Studio and to modify the **swagger.sh** and **serve.py** files.

**Figure 24**: Swagger.json, swagger.sh and serve.py
<img src="img/swagger/swagger-1.png" width="800">

Then we run the **swagger.sh** and **serve.py** files to be able to interact with the swagger instance running with the documentation for the HTTP API of the model.

**Figure 25**: Running serve.py

<img src="img/swagger/swagger-2.png" width="800">

This is the content of the API, diplaying the methods used to interact with the model.

**Figure 26**: API documentation - part 1
<img src="img/swagger/swagger-3.png" width="800">

**Figure 27**: API documentation - part 2
<img src="img/swagger/swagger-4.png" width="800">


### Consume model endpoints
Finally, it was time to interact with the model and feed some test data to it. We do this by providing the **scoring_uri** and the **key** to the **endpoint.py** script and running it.

After updating both the **scoring_uri** and the **key** to match the key for my service and the URI that was generated after deployment, I ran the **endpoint.py** script to get inference from the deployed model.

**Figure 28**: Updating endpoint.py
<img src="img/consume/consume-1.png" width="800">

**Figure 29**: Running endpoint.py
<img src="img/consume/consume-2.png" width="800">

#### (Optional) Benchmark
To do this, I made sure **Apache Benchmark** was installed and available. 

I need to update the **benchmark.sh** script.

**Figure 30**: Updating benchmark.sh
<img src="img/benchmark/benchmark-1.png" width="800">

After executing the **endpoint.py** script, we run the **benchmark.sh** script to load-test our deployed model.

**Figure 31**: Results after benchmark.sh execution
<img src="img/benchmark/benchmark-2.png" width="800">

As we can see, we got insights like Requests per second, Average time per request, Number of failed requests, etc.

### Create and publish a pipeline
For this step, I used the **aml-pipelines-with-automated-machine-learning-step.ipynb** Jupyter Notebook to create a **Pipeline**.

I created, consumed and published the best model for the bank marketing dataset using AutoML with Python SDK.

**Figure 32**: Pipeline running with Python SDK
<img src="img/pipeline/pipeline-1.png" width="800">

After updating the notebook to have the same keys, URI, dataset, cluster, and model names already created, I run through the cells to create a pipeline.

This is the pipeline created in the Pipelines section of Azure ML Studio.

**Figure 33**: Pipeline running in Azure Studio
<img src="img/pipeline/pipeline-2.png" width="800">

This is the Pipeline Overview in the Azure ML Studio.

**Figure 34**: Pipeline Overview in Azure Studio
<img src="img/pipeline/pipeline-4.png" width="800">
s
This is the **REST endpoint** in Azure ML Studio, with a status of ACTIVE.

**Figure 35**: The rest endpoint
<img src="img/pipeline/pipeline-3.png" width="800">

## Screen Recording

See project in action: <a href="https://youtu.be/GXZqAYFzaJ8">Link<a>

## Standout Suggestions
I think that I could be better if I would do feature engineering. Maybe It could be added to the pipeline in order to get better results.

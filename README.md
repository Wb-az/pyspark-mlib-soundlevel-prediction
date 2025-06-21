# MLib PySpark SoundLevel Prediction
Creates an ML Pipeline leveraging PySpark SQL and PySpark MLib to predict airfoils noise levels. 

This repository contains Data Engineering and ML material completed to obtain the [Data Engineering Professional Certificate](https://www.credly.com/badges/e0a1b78c-db53-484c-8363-2da95ba0582a/public_url) and the
[Machine Learning with Apache Spark Certification](https://www.credly.com/badges/886943d9-1bda-45be-a3eb-655a1cf98822/public_url) offered by IBM.                                                                                                     
                                                                                                               
The project aims to create an ML Pipeline from a data engineer perspective at an aeronautics consulting company
that designs airfoils for use in planes and sports cars. In this project, the                                  
[NASA Airfoil Self Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) dataset was used.        
The dataset was cleaned, and a data ML pipeline was created to evaluate the model that will predict the        
SoundLevel-based in multivariate data. The models were evaluated and persisted.                                
                                                                                                               
                                                                                                               
While the dataset and some parts of the ML pipeline structure are preserved, the notebook here is updated to work with ```PySpark version 3.5.3```.
The pipeline can be run using ```Final_Project.ipynb``` or ```main.py``` on the terminal.          
                                                                                                               
## Current Content                                                                                             
- Jupyter Notebook - ```Final_Project.ipynb```                                                                 
  - Processing data with Spark dataframes                                                                      
  - Variables correlation visualisations                                                                       
  - Training 4 Regression Models from PySpark MLib: Linear Regression, Factorized Machines, Random Forest and Gradient Boosted Regressor                                                      
  - Cross Validation for the best model from previous models                                                   
- Pure Python                                                                                                  
  - ```spark_utils.py``` - contains functions to build the pipeline                                            
  - ```main.py``` - runs the pipeline and saves the metrics in a parquet format
 
## 1. Variables Correlations                                               
                                                                           
![output.png](bigdata/output.png "Correlation Matrix")                             
**Fig 1**. Correlations heatmap for the NASA Airfoil Self Noise dataset.  

The AngleOfAttack and SuctionSideDisplacement variables are highly correlated (0.75) (Fig. 1).
                                                                                                               
## 2. Results                                                                                                  
                                                                                                               
### 2.1 Comparing regressors model with data training, validation and test splits approach                     
                                                                                                               
**Table 1**: Validation scores                                                                                 
                                                                                                               
| Metrics | Linear Regression | Factorized Machines | Random Forest | Gradient Boosted |                       
|---------|-------------------|---------------------|---------------|------------------|                       
| MAE     | 4.02              | 49.09               | 2.43          | 1.71             |                       
| MSE     | 26.38             | 3292.36             | 9.45          | 5.82             |                       
| RMSE    | 5.14              | 57.41               | 3.07          | 2.41             |                       
| R2      | 0.42              | -70.9               | 0.79          | 0.87             |                       
                                                                                                               
                                                                                                               
**Table 2**: Test scores                                                                                       
                                                                                                               
 | Metrics | Linear Regression | Factorized Machines | Random Forest | Gradient Boosted |                      
 |---------|-------------------|---------------------|---------------|------------------|                      
 | MAE     | 3.92              | 45.72               | 2.43          | 2.05             |                      
 | MSE     | 23.98             | 2980.4              | 10.02         | 8.21             |                      
 | RMSE    | 4.9               | 54.59               | 3.17          | 2.87             |                      
 | R2      | 0.57              | -52.49              | 0.82          | 0.85             |                      
                                                                                                               
It can be observed from Tables 1-2 that the best model is the Gradient Boosted Regressor.                                                                                                               
### 2.2 Fine-tune the best Cross Validation and Parameters Grid search - 5 Folds                               
                                                                                                               
                                                                                                               
**Table 3**: Best Model result and parameters for the Gradient Boosted Regressor optimisation                  
                                                                                                               
 | Model            | No Folds | RMS  | R2   | MaxIter | MaxDepth | MaxBins |                                  
 |------------------|----------|------|------|---------|----------|---------|                                  
 | Gradient Boosted | 5        | 2.60 | 0.87 | 25      | 8        | 25      |                                  
                                                                                                               
                                                                                                               
                                                                                                                                                                                                                  
 ### Notes:                                                                                                    
                                                                                                               
**Install the dependencies**                                                                                   
                                                                                                               
  - ```pip install findspark```                                                                                
  - ```pip install pyspark```                                                                                  
                                                                                                               
**Running in the terminal**                                                                                    
                                                                                                               
- ```python3 bigdata/main.py```                                                                                
                                                                                                               
                                                                                                               
                                                                                                               
                                                                                                               
                                                                                                               

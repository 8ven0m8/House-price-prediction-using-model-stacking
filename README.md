# Titanic Passenger Survival Prediction

![King county Houses](https://www.kcha.org/portals/0/Images/Leader/Leader_91.jpg)

## Overview

This project uses 3 Machine Learning algorithms stacked together to predict the house prices in King County, USA. Implemented feature engineering, selected parameters using hyperparameter optimization and handled extreme value outliers with IRQ(Interquartile Range) method.

Visit kaggle notebook :- https://www.kaggle.com/code/pranjalsapkota/house-price-prediction-using-model-stacking

#### The stack's MAE and MAPE is 48.7k and 11.05% respectively.

## Dataset

The dataset used in this project contains house sale prices of King county, USA.
https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/data

Typical features in the dataset include:

* id                             
* date             
* price            
* bedrooms         
* bathrooms        
* sqft_living      
* sqft_lot         
* floors           
* waterfront       
* view             
* condition        
* grade            
* sqft_above       
* sqft_basement    
* yr_built         
* yr_renovated     
* zipcode          
* lat              
* long             
* sqft_living15    
* sqft_lot15       

Target variable:

* **price**

## Project Workflow

1. Environment Setup
2. Load Dataset
3. Initial Data Cleaning
4. Handling outliers
5. Feature engineering
6. Split Data
7. Hyperparameter Optimization
8. Training Models
9. Analyzing Feature Importance
10. Model Evaluation
11. Test Input
12. Plot for Actual vs Predicted price


## Model Used

* Random Forest Regressor
* XGBoost Regressor
* LightGBM Regressor

## Example Prediction Input

Example house data:

* house_no = 1

The model predicts the house price, compares to actual price and gives the error amount.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Jupyter Notebook

## References

* Kaggle House Sales in King County, USA Dataset
* Scikit-learn Documentation


# CreditRiskAnalysis
Here I tried to do Analysis about Credit Risk. This project can be used to know the characteristics of creditors and to predict the probability of future defaults and credit card borrowings

## Data Understanding
The dataset is about Credit and personal information dataset by Seanny from Kaggle and licensed for its author to use publicly. The goal of this project is to predict whether a client will become a defaulter. Personal detail and credit record for each client are presented in the dataset.

## Data Modelling Outline
The following are a few steps in the modeling technique used on this project:
1. Import dataset
2. Data understanding
3. Data cleaning & preprocessing
4. EDA (Exploratory Data Analysis)
5. Feature Engineering
6. Machine learning modeling
7. Conclusion

## Load the Dataset
The raw dataset must be combined before proceeding with the process. For dataset with personal information of each client, There are a total of 438,557 rows and 18 columns. Each column is labeled with a description of its own. For dataset with credit record of each client, There are a total of 1,0485,75 rows and 3 columns.

## Data Understanding
Below is the description for each column or feature:
* ID: Client's ID
* CODE_GENDER: Gender of Clients
* FLAG_OWN_CAR: Does the client own a car?
* CNT_CHILDREN: Client's children count
* AMT_INCOME_TOTAL: Client's annual income
* NAME_INCOME_TYPE: Client's income type
* NAME_EDUCATION_TYPE: Client's education level
* NAME_FAMILY_STATUS: Client's family status
* NAME_HOUSING_TYPE : Client's housing type
* DAYS_BIRTH: The days the client was born, counts backwards (-1000 means client was born 1000 days ago)
* DAYS_EMPLOYED: The days the client was employed, counts backwards (-1000 means client was employed 1000 days ago, 0 means unemployed)
* FLAG_MOBIL: Does client have mobile phone?
* FLAG_WORK_PHONE: Does client have work phone?
* FLAG_PHONE: Does client have phone?
* FLAG_EMAIL: Does client have email?
* OCCUPATION_TYPE: Client's occupation
* CNT_FAM_MEMBERS: Client's family members count
* MONTHS_BALANCE: Client's record month
* STATUS: Default or not.

## Data cleaning & preprocessing
Data cleaning is used to identify data that has a NaN value so that it can be processed further using machine learning modeling. The preprocessing process is used to convert the value of a specific column into the appropriate value for modeling purposes. Data preprocessing will therefore include a data cleaning process to detect any missing values in each column. There are missing value in feature 'OCCUPATION_TYPE' because the missing value's percentage is quite big and we don't have enough info the missing value will be replaced by 'unknown'.

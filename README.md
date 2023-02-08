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

## Exploratory Data Analysis (EDA)
* How is the characteristics of each client? <br>
After plotting each features we got some interesting insight. <br>
1. Most client don't have any children which is around 63.8% of total client
<img src="https://user-images.githubusercontent.com/89758536/217485177-05f1074b-1f15-4df2-9b2d-ec7c887c0a52.png" width="500">
2. Most client has income type of working which is around 52% of total client
<img src="https://user-images.githubusercontent.com/89758536/217485305-ddef91b1-7d7e-42fc-8d99-08e9c1c43afe.png" width="500">
3. Most client's education are on Secondary special which is around  70% of total client
<img src="https://user-images.githubusercontent.com/89758536/217485608-8c3eb13e-1a19-4f7e-9437-384a504c9edd.png" width="500">
4. Most client has house / apartement.
<img src="https://user-images.githubusercontent.com/89758536/217485757-63615e77-783c-42e4-b58c-73f6d7c3edbe.png" width="500">
5. 30% of the client has unknown occupation. But the second highest occupation is Laborers which around 17% of client
<img src="https://user-images.githubusercontent.com/89758536/217485844-227bdda9-3243-4d8b-9c07-85a4001207f1.png" width="500">
6. Most clients are around age of 31-45 years old
<img src="https://user-images.githubusercontent.com/89758536/217486043-5cbafc3a-72da-49bf-823b-896d31ca80f7.png" width="500">
7. Most clients dominated by people who had been working for 0-10 years
<img src="https://user-images.githubusercontent.com/89758536/217486425-d70b18c5-53c5-425e-85ae-737480e036e6.png" width="500">

* How is the characteristics of clients who become defaulters
1. The more family members the percentage of the trend of default client is increasing
<img src="https://user-images.githubusercontent.com/89758536/217486741-06b83085-902b-4554-9438-4c5b036d225a.png" width="500">
2. State servant and Commercial associate has the highest annual income compared with the others but has the higher percentage of default client which are around 12.9% and 12.72% of them are default.
<img src="https://user-images.githubusercontent.com/89758536/217487221-a59ca7dd-f04c-4463-8e45-be2d295da9fd.png" width="500">
<img src="https://user-images.githubusercontent.com/89758536/217488052-595bc09c-d33c-42ce-ae68-05c620637969.png" width="500">
3. Academy education contributes the most default client compared among other education level which is around 21.88%
<img src="https://user-images.githubusercontent.com/89758536/217488414-77617808-97b2-4b6c-8df4-5fef7678acc8.png" width="500">
<img src="https://user-images.githubusercontent.com/89758536/217488320-9884ce6d-88c1-4efa-a4ef-5334f894f2ad.png" width="500">
4. Client who lives in office apartment has the highest risk among other housing type which is around 14.5% even though they have the highest annual income compared with the others
<img src="https://user-images.githubusercontent.com/89758536/217488613-cace3c84-2c2e-40a1-b420-6fcbc4da8163.png" width="500">
<img src="https://user-images.githubusercontent.com/89758536/217488613-cace3c84-2c2e-40a1-b420-6fcbc4da8163.png" width="500">
5. Most default clients are dominated by clients around 20 – 35 years old
<img src="https://user-images.githubusercontent.com/89758536/217488903-a02761ea-32e9-4d04-b7d9-9abd47a020cd.png" width="500">
6. Most default clients are dominated by clients who has employment years of 31-40 years
<img src="https://user-images.githubusercontent.com/89758536/217489093-25613665-2535-432a-9cb5-d52d4f090c5b.png" width="500">

* At what period does the client tend to default more?
<img src="https://user-images.githubusercontent.com/89758536/217489230-13dcfde2-6546-4321-8e38-3515d8e5815b.png" width="700">
Here I took the credit info of clients based on various time period. We could see that the percentage of customer who had credit for 0-15 months tends to increasing rapidly. After that period the increment is starting to increase steadily.

## Feature Engineering
Feature engineering is used to create new features such as new columns and dummy variables to make the data more precise. The goal of feature engineering is to create a new set of columns that will produce more accurate results for machine learning modeling. For feature engineering, we're going to create new columns that will determine the age and employment years of client. We will also groupin credit status of client into two categories which are default and not-default. Default means the client has been late to pay their payment for more than 60 days. To change certain columns into more appropriate values, label encoding and one hot encoding must be used. I also treated the imbalance data handling using Undersampling method. And last I scaled the data using StandardScaler.

## Machine Learning Modeling
* This model used Principal Component Analysis (PCA) to reduce the total features from 56 features to 25 features
* The dataset split with 75:25 ratio which means 75% data become training data and the remaining become test data.
* As the metric, we chose ‘Recall’ as the metric because in credit risk analysis, we want to minimize the false negative scenario where we predict the customer won’t be default but the actual is they’ll become default. It could cause the company to loss profit more.
* The best algorithm to use is K-Nearest Neighbors with the value of recall is 86%
* From the confusion matrix we can analyze that the model has True Positive Rate of 86% means from 2915 clients that our model predicts will become defaults, 86% of them or  2502 of them will exactly be default!!.<br>
![image](https://user-images.githubusercontent.com/89758536/217491189-4e111a66-8e31-4d1f-a1c9-d4a28f40446d.png)

## Machine Learning Impact to Business
* Let’s assume that there are 1000 clients who are labeled as default customer. Using this model, we could make sure that from those 1000 clients. 860 of them really are default customer. Let’s say for each default customer the company has a loss of $10. It means the company can save up to $8,600.00 by using this model!.

## Conclusion & Recommendation
1. There are some dominant characteristics among the clients which are have no children, who are working, have secondary education, have house or apartment, are laborers, around age 31-45 years old and have been in employment for around 0-10 years. The marketing team can focus their effort into promoting advertisement into this group.
2. There are some dominant characteristics among the clients who are default clients which are the more family members they have the more they become default, who are state servant, have academic degree, and are around 20-35 years old. The approval team can keep their eyes on this particular group and check their record closely to make sure could they will not become default client.
3. Annual doesn’t always positively correlated with probability of becoming default client. For example, as we can see in income type, education level, and housing type the highest category that has higher annual income tend to have bigger percentage of default client.
4. From vintage analysis we could see that the percentage of client becoming default is increasing rapidly around 0-15 month of the ongoing credit. After that, the percentage of client becoming default is increasing steadily. The approval can use this insight to approve people who has crediting for more than 15 month more easily. And team marketing can also focus more on this group to advertise their services.
5. The best model in this case is KNN which gives recall around 86%. This model can be used in an application in which approval team could insert the personal information and credit information of clients who applied, then this model could predict would this client be default or not and from there it can be consideration to approve the client or not.


For more detail, Feel free to see my code







## Data Science Project Classification : Telcom Churn Classification Overview
* Build a machine learning model to predict and classify wether a certain Telcom customers would churn from internet / broadband services.
* Telcom customer records data of over 7000 ID's / customers.
* Feature engineered and transformation for dataset to better be read and be implemented to model.
* Data Exploration gives us better understanding which any of the criterias leading up to customer end subscribing to telcom's internet / broadband service.
* Using several classifier models to determine which features contributes the most in terms of customer churn.

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index1.png) <br>

Customer attrition (a.k.a customer churn) is one of the biggest expenditures of any organization. If we could figure out why a customer leaves and when they leave with reasonable accuracy, it would immensely help the organization to strategize their retention initiatives manifold. Let’s make use of athis Telco customer transaction dataset from to understand the key steps involved in predicting customer attrition. <br>

Supervised Machine Learning is nothing but learning a function that maps an input to an output based on example input-output pairs. A supervised machine learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. Given that we have data on current and prior customer transactions in the telecom dataset, this is a standardized supervised classification problem that tries to predict a binary outcome (Y/N). <br>

Defining business problems <br>
### Business Problems :
 * What is the probability of a customer to stop / churning from using the services?
 * Which factors play the most role when it comes to churn?
 * WHat business actions needed to be taken to addres / minimize this issue?

### Machine Learning Frame work step by step:
* Dataset Profiling
* Data Cleaning if theres any missing value
* Visualization 
* Pre processing dataset
* Model selection and training
* Performaces Evauation

### Code and Resources Used
* **Packages** : pandas, numpy, matplotlib, seaborn, sci-kit learn, shap, yellowbrick, lightgbm.

### Dataset Profiling
* Dataset consists of 7043 observations and 23 columns with a total dataset size of 147903.
### Features Types
 * Majority of features fall on categorical one, meanwhile the rest of 3 are numerical. Notable numerical ones include, TotalCharge, Monthly Charge and tenure.
 
### Data Cleaning
![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index1.png) <br>

* Missing values were non-existent in data, however i had to do some data cleaning and transformation for `TotalCharges`. Some of its values were left blank. For this, I replcaed those values with zero (0). On top of transforming thius object types data to its proper type, numerical.
* Feature transfomation of `Seniorcitizen`, replacing its initial distinct value of both '0' and '1' to to 'Yes" and 'No'. And transform it to object type.
* Cross Checking if dataset had properly been cleaned. <br>

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index1.png)

### Descriptive Statistics

* A quick look at the statistics suggests that, on average telco customers are staying for 32 months and are paying $64 per month. However, this could potentially be because different customers have different contracts. From the looks of things, we can presume that dataset contains several features which will provide an various informations about the customer transation. 

### Features Distribution

### Boxplot Plotting

### Exploratory Data / Visualization

### Frequency Distribution of Churn per PhoneService and InternetService

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index1.png)

* As we can see, customers who subscibe to DSL type of internet service are more likely to to not churn, while the opposite is true. This is evident by the high number of customers distribution in that regards. This is also true for those customers who are at the same time using the phone service on top of internet service will also more likely to continue using the services.

### Frequency Distribution of Churn per StreamingTV and StreamingMovies

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index2.png)

* Customers who are subcribing to both service of Movies and TV streaming are mmore likely to not churn from the services. Evidence by the high number of customers count distribution in that regards. Meanwhile, for customers who are only subscribing to either of said services wont likely to prolong their subscription.

### Frequency Distribution of Churn per PaymentMethod and PaperlessBIlling

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index3.png)

* As for how customers paying their annual subscription fee, the ones who more likey to prolong their subscription are those paying with either of this 3 methods, Mailed Check, Bank Transfer and Credit Card. On the other hand, customers who churn from subscription are dominated by the ones who pay their annual fee by using Electronic Check. 

### Frequency Distribution of Churn per Contract

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index4.png)

* Its clear in regards to subscription contract,  customers who are more likely to churn from subsciption are those who only agree upon month by month basis contract subscription. It is understood, usually for customers like them, they are doing trial month before deciding to use Telcom's services.
* 
### Data Preparation Before Modeling

* Imbalanced target feature `Churn` checking. This is to ensure which scoring parameter is best used for dataset.
* Train and Test splitting with a proportion of 75% Train and 20% Test.
* Standardizing for numerical data using Robust scaler and Onehot encoder for categorical features.

### Model Building
* Tree-based algorithm model were used as they can plot features importances and gives weight to features and determines which ones are contribute leading up to customer churn.
* 3 Model used are `Random Forest, Stochastic Gradient Boosting, LightGBM`.

### Model Performances
* **Random Forest**
```
*****************Train*******************
Train Accuracy 0.9979174555092768
Train Precision 0.9971509971509972
Train Recall 0.9950248756218906
Train F1 Score 0.9960868018498755
****************************************
******************Test*******************
Test Accuracy 0.7881885292447472
Test Precision 0.6219178082191781
Test Recall 0.49134199134199136
Test F1 Score 0.5489721886336155
*****************************************
```

* **Stochastic Gradient Boosting**
```
*****************Train*******************
Train Accuracy 0.8330177962892844
Train Precision 0.7312775330396476
Train Recall 0.589907604832978
Train F1 Score 0.6530291109362707
****************************************
******************Test*******************
Test Accuracy 0.8012492901760363
Test Precision 0.6414141414141414
Test Recall 0.5497835497835498
Test F1 Score 0.592074592074592
*****************************************
```

* **LightGBM**
```
*****************Train*******************
Train Accuracy 0.8835668307459296
Train Precision 0.8245901639344262
Train Recall 0.7149964463397299
Train F1 Score 0.7658926532165969
****************************************
******************Test*******************
Test Accuracy 0.7950028392958546
Test Precision 0.6265664160401002
Test Recall 0.5411255411255411
Test F1 Score 0.5807200929152148
*****************************************
```

* Amongst the 3 model tested, `Stochastic Gradient Boosting`model has the edge over the rests, indicated by its higher number of scores like `Accuracy, F1, Precision, Recall`. Will be using this model as a base to build an optimized model further improving the scores. 
* The Optimized model were build upon the SGB model. Optimization are done to reduce basically non performing features, features which contribute less or even dont contributes at all to prediction scores. The value of these can bee seen using model integrated's feature importance. 

### Best Model Confusion Matrix and Classificatio Reports
* Confusion Matrix

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index.png)

* Classification Reports
```
                precision    recall  f1-score   support

           0       0.85      0.89      0.87      1299
           1       0.64      0.55      0.59       462

    accuracy                           0.80      1761
   macro avg       0.74      0.72      0.73      1761
weighted avg       0.79      0.80      0.80      1761
```





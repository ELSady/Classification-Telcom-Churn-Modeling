## Data Science Project Classification : Telcom Churn Classification Overview
* Build a machine learning model to predict and classify wether a certain Telcom customers would churn from internet / broadband services.
* Telcom customer records data of over 7000 ID's / customers.
* Feature engineered and transformation for dataset to better be read and be implemented to model.
* Data Exploration gives us better understanding which any of the criterias leading up to customer end subscribing to telcom's internet / broadband service.
* Using several classifier models to determine which features contributes the most in terms of customer churn.

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/1%20S1cWdTfyeF-Rp64ht1rkuQ.jpeg) <br>

Customer attrition (a.k.a customer churn) is one of the biggest expenditures of any organization. If we could figure out why a customer leaves and when they leave with reasonable accuracy, it would immensely help the organization to strategize their retention initiatives manifold. Let’s make use of athis Telco customer transaction dataset from to understand the key steps involved in predicting customer attrition. <br>

Supervised Machine Learning is nothing but learning a function that maps an input to an output based on example input-output pairs. A supervised machine learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. Given that we have data on current and prior customer transactions in the telecom dataset, this is a standardized supervised classification problem that tries to predict a binary outcome (Y/N). <br>

Defining business problems <br>
### Business Problems :
 * What is the likelihood of a customer to stop / churning from using the services?
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
* Dataset consists of 7043 observations and 23 columns with a total dataset size of 147903. <br>

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/Screenshot%202022-06-26%20at%2015-09-19%20Churn%20Classification%20-%20Jupyter%20Notebook.png) <br>

### Features Types
 * Majority of features fall on categorical one, meanwhile the rest of 3 are numerical. Notable numerical ones include, TotalCharge, Monthly Charge and tenure.<br>

![alt text]([https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index1.png](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/Screenshot%202022-06-26%20at%2015-10-01%20Churn%20Classification%20-%20Jupyter%20Notebook.png)) <br>
 
### Data Cleaning

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/Screenshot%202022-06-26%20at%2015-12-11%20Churn%20Classification%20-%20Jupyter%20Notebook.png) <br>

* Missing values were non-existent in data, however i had to do some data cleaning and transformation for `TotalCharges`. Some of its values were left blank. For this, I replcaed those values with zero (0). On top of transforming thius object types data to its proper type, numerical.
* Feature transfomation of `Seniorcitizen`, replacing its initial distinct value of both '0' and '1' to to 'Yes" and 'No'. And transform it to object type.
* Cross Checking if dataset had properly been cleaned. <br>

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/Screenshot%202022-06-26%20at%2015-12-40%20Churn%20Classification%20-%20Jupyter%20Notebook.png) <br>

### Descriptive Statistics

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/Screenshot%202022-06-26%20at%2015-12-57%20Churn%20Classification%20-%20Jupyter%20Notebook.png) <br>

* A quick look at the statistics suggests that, on average telco customers are staying for 32 months and are paying $64 per month. However, this could potentially be because different customers have different contracts.

### Features Unique Value Checking
* ‘Payment Methods’ and ‘Contract’ are the two categorical variables in the dataset. When we look into the unique values in each categorical variables, we get an insight that the customers are either on a month-to-month rolling contract or on a fixed contract for one/two years. Also, they are paying bills via credit card, bank transfer or electronic checks.

### Numerical Features Distribution

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index.png)

Insights we can made from the distribtion plot:
* Majority of customer are paying total charges for the services of around 300 to 500
* Whilst for monthly charges they have to pay on average is around 25 to 30
* It is also inferred that there are many new customers (less than 10 months old) and many loyal customers (more than 70 months old), the rest are in between those two categoriews.

### Boxplot Plotting

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index2.png)

### Exploratory Data / Visualization

### Frequency Distribution of Churn per PhoneService and InternetService

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index3.png)

* As we can see, customers who subscibe to DSL type of internet service are more likely to to not churn, while the opposite is true. This is evident by the high number of customers distribution in that regards. This is also true for those customers who are at the same time using the phone service on top of internet service will also more likely to continue using the services.

### Frequency Distribution of Churn per StreamingTV and StreamingMovies

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index4.png)

* Customers who are subcribing to both service of Movies and TV streaming are mmore likely to not churn from the services. Evidence by the high number of customers count distribution in that regards. Meanwhile, for customers who are only subscribing to either of said services wont likely to prolong their subscription.

### Frequency Distribution of Churn per PaymentMethod and PaperlessBIlling

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index5.png)

* As for how customers paying their annual subscription fee, the ones who more likey to prolong their subscription are those paying with either of this 3 methods, Mailed Check, Bank Transfer and Credit Card. On the other hand, customers who churn from subscription are dominated by the ones who pay their annual fee by using Electronic Check. 

### Frequency Distribution of Churn per Contract

![alt text](https://github.com/ELSady/Classification-Telcom-Churn-Modeling/blob/main/index6.png)

* Its clear in regards to subscription contract,  customers who are more likely to churn from subsciption are those who only agree upon month by month basis contract subscription. It is understood, usually for customers like them, they are doing trial month before deciding to use Telcom's services.

### Data Preparation Before Modeling

* Imbalanced target feature `Churn` checking. This is to ensure which scoring parameter is best used for dataset.

* It is confirmed. dataset is ont imblaanced, so instead of accuracy, ROC / AUC score will be the propr evaluation metric of models 
* Preprocessing data with pycaret with the following parameters:


### Building and Evaluating Models Performances

* To do a quick comparison and evaluation to models, we ussualy want to check the `Accuracy` score. Classification accuracy itself is a metric that summarizes the performance of a classification model as the number of correct predictions divided by the total number of predictions.
* However because noted, our dataset is not an imbbalanced one, so the appropriate metric to evaluate can be either `ROC` or `AUC` score that is otherwise will be a misleading should we use the Accuracy metric.
*  AUC - ROC curve is a performance measurement for the classification problems at various threshold settings. ROC is a probability curve and AUC represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. Higher the AUC, the better the model is at predicting 0 classes as 0 and 1 classes as 1. By analogy, the Higher the AUC, the better the model is at distinguishing between customer which will curn and not churn.
* So, based on the AUC score here we have the best 3 models used as a base. Logistic Regression, Adaboost and Gradient Boosting with a very good AUC score of around 0.85.

### Feature Importances 
* Taking a look at the featuer importances of the 3 models. Tenure and Total Charges and contract types play a huge role to determine the likelihood of a customer to churn or keep using the services provided by telco.
* Recall back on our visualization, it is indeed correct, customers who likely to hurn are those who ona short period time tenure hence the low charges count, and the oppsite is true, ones who have been a customer for a long time have a higher charges count.
* The same applies to contract types, customers who are more likely to churn are those who only agree upon month by month basis contract subscription.
* Seems the model is good fit to dataset.

### Confusion Matrix Model Evaluation
* A confusion matrix is a table that is used to define the performance of a classification algorithm. A confusion matrix visualizes and summarizes the performance of a classification algorithm. 


* Confusioin matrix above shows the model (ADAboost) have a total of 1375 + 301 correct predicitons while it also has 260 + 177 wrong predictions.

### Predicition on unseen data

* Impressive, the models we used (ADAboost) still retain a very good score of AUC with 0.85 on predicting the unseen dataset. This is the model we are going to use for classifiying customer churn.

### Conclusion 
* Customers with a month-to-month connection have a very high probability to churn that too if they have subscribed to pay via electronic checks.
* Many of the customer are younger ones.
* There are a lot of new customers in the organization (less than 10 months old) followed by a loyal customer base that’s above 70 months old.
* 

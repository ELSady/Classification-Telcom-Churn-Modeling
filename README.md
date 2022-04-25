## Data Science Project Classification : Telcom Churn Classification Overview
* Build a machine learning model to predict and classify wether a certain Telcom customers would churn from internet / broadband services.
* Telcom customer records data of over 7000 ID's / customers.
* Feature engineered and transformation for dataset to better be read and be implemented to model.
* Data Exploration gives us better understanding which any of the criterias leading up to customer end subscribing to telcom's internet / broadband service.
* Using `Tree-based algorithm's feature importance` to determine which features contributes the most in terms of customer churn.
* Optimized Tree-based algorithm to output higher score `Accuracy, F1, Precision, Recall`.

### Code and Resources Used
* **Packages** : pandas, numpy, matplotlib, seaborn, sci-kit learn, shap, yellowbrick, lightgbm.

### Data Cleaning
* Missing values were non-existent in data, however i had to do some data cleaning and transformation for `TotalCharges`. Some of its values were left blank. For this, I replcaed those values with zero (0). On top of transforming thius object types data to its proper type, numerical.
* Feature transfomation of `Seniorcitizen`, replacing its initial distinct value of both '0' and '1' to to 'Yes" and 'No'. And transform it to object type.
* Cross Checking if dataset had properly been cleaned.

### Exploratory Data Highlight

### Frequency Distribution of Churn per PhoneService and InternetService

### Frequency Distribution of Churn per StreamingTV and StreamingMovies

### Frequency Distribution of Churn per PaymentMethod and PaperlessBIlling

### Frequency Distribution of Income per Contract

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

### Best Model Confusion Matrix and Classificatio Reports

* Classification Reports
```
                precision    recall  f1-score   support

           0       0.85      0.89      0.87      1299
           1       0.64      0.55      0.59       462

    accuracy                           0.80      1761
   macro avg       0.74      0.72      0.73      1761
weighted avg       0.79      0.80      0.80      1761
```





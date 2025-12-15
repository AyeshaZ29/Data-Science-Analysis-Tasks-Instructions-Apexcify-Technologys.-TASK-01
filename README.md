•	Kaggle link
https://www.kaggle.com/datasets/uciml/iris?resource=download

Complete Overview
This file performs a full data analysis + machine learning workflow on the Iris dataset.

It goes far beyond basic statistics and includes EDA, data cleaning, visualization, and multiple ML models. 

1. Dataset Loading
•	Imported Pandas
•	Loaded Iris dataset from a CSV file
Dataset successfully loaded for analysis.

3. Basic Descriptive Statistics (TASK 1 requirement)
The following statistics were calculated:
•	describe() → count, mean, std, min, max, quartiles
•	Mean
•	Median
•	Minimum
•	Maximum
•	Standard deviation
•	Missing values using isnull().sum()
✔ Result: No missing values found in the dataset.


ADDITIONAL TASK
Data Inspection
Used to understand dataset structure:
•	head() and tail()
•	info() (data types & nulls)
•	columns
•	value_counts() for Species
✔ Verified dataset has 150 rows and 5 columns.
5. Data Cleaning
Performed cleaning operations:
•	Checked and removed missing values (dropna)
•	Removed unnecessary Id column
•	Checked duplicates
•	Verified class balance
•	Ensured clean dataset
✔ Final dataset:
150 rows × 5 columns (4 numeric + 1 categorical target) 

6. Exploratory Data Analysis (EDA)
Visual analysis using Matplotlib & Seaborn:
Visuals created:
•	Histograms (feature distributions)
•	Boxplots (feature vs species)
•	Scatter plots
•	Pair plots
•	Correlation heatmaps
•	Count plots
✔ Key insight:
Petal length & petal width are the strongest features for species classification.

8. Feature Encoding
•	Converted categorical Species into numeric form using LabelEncoder

✔ Data prepared for machine learning.

8. Train–Test Split
•	Features → X
•	Target → Y
•	80% training, 20% testing
train_test_split(X, Y, train_size=0.80)
9. Machine Learning Models Implemented
The file trains and evaluates multiple classifiers:
Models used:
	Logistic Regression
	Decision Tree
	Gaussian Naive Bayes
	Support Vector Classifier (SVC)
	Random Forest Classifier
  
For each model:
•	Accuracy calculated
•	Confusion matrix plotted
•	Classification report printed
•	Feature importance / coefficients visualized
✔  Performance comparison of different algorithms on Iris dataset.

9. Model Evaluation
Used:
•	accuracy_score
•	confusion_matrix
•	classification_report
•	Heatmaps for better interpretation
✔ Helped analyze precision, recall, f1-score per class.


Final Summary
•	This file loads the Iris dataset, performs descriptive statistics, cleans the data, conducts full EDA, and applies multiple machine learning models to classify Iris species with performance evaluation

# Binary Classification using Multiple Machine Learning Models

## Project Overview
This project presents a comparative analysis of multiple machine learning classifiers on a binary classification dataset consisting of 300 samples. The objective is to evaluate and compare the performance of linear and non-linear models under the same experimental setup.

## Dataset
- Total samples: 300  
- Task: Binary classification  
- Features: Numerical attributes describing product characteristics  
- Target variable: Binary class label  

The dataset is included in this repository as `Product Data.csv`.

## Methodology

1. Data loading and preprocessing  
2. Train-test split  
3. Model training  
4. Performance evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score  

## Implemented Models

- Logistic Regression  
- Linear SVM  
- RBF SVM  
- Random Forest  
- K-Nearest Neighbors  
- Naive Bayes  

All models were evaluated under the same data split for fair comparison.

## Results Summary

The models demonstrated varying behavior across evaluation metrics. Non-linear models (RBF SVM and Naive Bayes) achieved the highest accuracy on the test set. Linear SVM showed high recall but lower precision, indicating sensitivity to decision boundary structure.

## Error Analysis

Although some models achieved 100% accuracy on the test split, the relatively small dataset size (300 samples) may limit generalization. Comparative analysis across models highlights sensitivity to model complexity and class boundary structure, emphasizing the importance of robust validation strategies in small datasets.



## My Contribution

I independently designed and implemented the complete machine learning pipeline, including data preprocessing, model training, comparative evaluation, and performance analysis across multiple classifiers.

## Reproducibility

The project can be reproduced by running `ProductData_Code.ipynb`.  
Required libraries include:
- pandas
- numpy
- scikit-learn
- matplotlib (if used for visualization)

---

This project demonstrates practical implementation of classification algorithms, comparative evaluation methodology, and analytical reasoning in model assessment.

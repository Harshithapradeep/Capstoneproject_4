|Project Title|Microsoft : Classifying Cybersecurity Incidents with Machine Learning
| :--- | :--- |
|**Skills take away From This Project**| ●	Data Preprocessing and Feature Engineering●	Machine Learning Classification Techniques●	Model Evaluation Metrics (Macro-F1 Score, Precision, Recall)●	Cybersecurity Concepts and Frameworks (MITRE ATT&CK)●	Handling Imbalanced Datasets●	Model Benchmarking and Optimization
|**Domain**|Cybersecurity and Machine Learning|

<ins>**Project Description:**</ins>
This project aims to enhance the efficiency of SOCs by developing a machine learning model that can accurately predict the triage grade of cybersecurity incidents. Utilizing the comprehensive GUIDE dataset to create a classification model that categorizes incidents as true positive (TP), benign positive (BP), or false positive (FP) based on historical evidence and customer responses to implement the following 
- Security Operation Centres (SOCs): Automating the triage process by accurately classifying cybersecurity incidents, allowing analysts to respond to critical threats.
- Incident Response Automation: Enabling guided response systems to automatically suggest appropriate actions for different types of incidents, leading to quicker mitigation of potential threats.
- Threat Intelligence: Improved accuracy in identification of true and false positives.
- Enterprise Security Management: Enhanced security posture by reducing false positives

  <ins>**Data Exploration and Understanding:**</ins>
- Step 1: Initial Inspection
  Loading the train.csv dataset and perform an initial inspection to understand the shape of the dataframe ,check data types, duplicated records, Target variable distribution and  non-null record counts.
- Step 2: Exploratory Data Analysis 
Statistical summary of numerical data columns and correlations ,missing values ,detect outliers  and class imbalances

  <ins>**Data Preprocessing steps:**</ins>

- Handling Missing Data: Dropping columns with more than 80% missing values and Imputation strategies median for numerical and mode for categorical data columns.
- Outlier handling : Implemented Inter quartile range method to detect and remove outliers.
- Feature Engineering: Creating time date features from timestamp, normalizing data.
- Encoding Categorical Variables: One-hot and label encoding.
- Feature Scaling: Standardization to ensure equal contribution of features.

 <ins>**Data Splitting and Sampling:**</ins>

- Split data into training and validation sets for robust model evaluation.
- 80-20 train-validation split with stratified sampling.
- Outcome: Ensures balanced class distribution across training and validation sets

 <ins>**Model Selection and Training:**</ins>
Select and train models to classify incidents effectively.
- Baseline models:
1)	Logistic Regression: Linear model used for binary classification tasks.
- Performance: Achieved an accuracy of 88% with the following metrics:
- Precision: 0.77 (macro avg), 0.87 (weighted avg)
- Recall: 0.72 (macro avg), 0.88 (weighted avg)
- F1-score: 0.74 (macro avg), 0.87 (weighted avg)
- Use Case: Suitable for problems with linear relationships and scenarios where model interpretability is crucial. Limited effectiveness with class imbalance.

2)	Decision Tree: Non-linear model known for its easy interpretability.
- Performance: Achieved an accuracy of 94% with the following metrics:
- Precision: 0.86 (macro avg), 0.94 (weighted avg)
- Recall: 0.87 (macro avg), 0.94 (weighted avg)
- F1-score: 0.87 (macro avg), 0.94 (weighted avg)
- Use Case: Ideal for problems with non-linear relationships and smaller datasets where model interpretability is beneficial

Advanced models:

1)	Random Forest: Ensemble model combining multiple decision trees to improve predictive performance.
- Training Process: 5-fold cross-validation; hyperparameter tuning for optimal model configuration.
- Performance: Achieved 98% accuracy, 96% macro-F1 score.
- Use Case: Effective for high-dimensional data and capturing complex feature interactions.

2)	Gradient XG Boost : Gradient boosting model optimized for speed and performance.

- Training Process: Hyperparameter tuning (learning_rate, max_depth), grid search.
- Performance: Achieved 98% accuracy, 95% macro-F1 score.
- Use Case: Ideal for structured data and often used in competition-winning solutions for tabular data.

3)	Light GBM : Gradient boosting framework designed for high efficiency and scalability.
- Training Process: Optimized for memory usage and speed, suitable for large datasets.
- Performance: Achieved 98% accuracy, 96% macro-F1 score.
- Use Case: Excellent for real-time predictions and processing large-scale datasets.

4)	Neural Networks: Deep learning model designed to capture complex, non-linear patterns in data.
- Architecture: Three hidden layers with dropout for regularization; learning rate tuning. 
- Performance: Achieved 88% accuracy, 77% macro-F1 score. 
- Use Case: Best for handling large datasets with non-linear relationships, including image and text data.

<ins>**Model Evaluation metrics:**</ins>

- Macro-F1 Score: Measures balanced performance across all classes by averaging the F1 Scores for each class.
- Precision (Macro): Evaluates the accuracy of positive predictions by focusing on minimizing false positives across all classes.
- Recall (Macro): Assesses the model’s ability to detect actual positives by focusing on maximizing true positive detection across all classes.

<ins>**Hyper parameter Tuning:**</ins>
- Random Search: An efficient method for hyperparameter optimization that samples from a predefined range of hyperparameters randomly, quicker than Grid Search and can find optimal settings by exploring different combinations of parameters.
- SMOTE (Synthetic Minority Over-sampling Technique): Addresses class imbalance by generating synthetic samples for the minority class. This technique helps improve model performance by providing a more balanced training dataset and reducing bias towards the majority class.

<ins>**Tuning Process – Insights:**</ins>
- Random Forest: Optimal parameters {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 0.75, 'max_depth': 50, 'bootstrap': False} improved the macro-F1 score to 92%. The model achieved 96% accuracy with strong precision and recall across all classes.
- Impact of Tuning: Enhanced precision and recall, especially for minority classes, with very few misclassifications in the confusion matrix. 
  - [[ 1607   107    31]
  - [  126  1823    46]
  - [  204    78 12881]]
- Performance: Achieved 96% accuracy, 92% macro-F1 score. 

<ins>**Model interpretation:**</ins>

Techniques Used:
- Feature Importance: Analyzing importance scores derived from Random Forest models.
- Error Analysis: Reviewing misclassified cases to identify potential areas for improvement
- Top Features:
  - Incident Id: 0.32
  - Org Id: 0.15
  - MITRE Techniques: 0.15
  - Alert Id: 0.08
  - Entity Type: 0.06
- Impact on Predictions:
Incident Id and Org Id significantly influence the model's classification, particularly in distinguishing between different classes.

<ins>**Error Analysis and Performance:**</ins>

- Misclassification Summary:
    - [[ 1607   107    31]
    - [  126  1823    46]
    - [  204    78 12881]]

- The off-diagonal elements are:
    - From the first row: 107+31=138
    - From the second row: 126+46=172
    - From the third row: 204+78=282
    - Total Misclassifications =138+172+282=592
- Total Misclassifications: 592
- Performance Metrics: Accuracy: 96%
- Classification Report:
    - Precision: 0.83 (Class 0), 0.91 (Class 1), 0.99 (Class 2)
    - Recall: 0.92 (Class 0), 0.91 (Class 1), 0.98 (Class 2)
    - F1-Score: 0.87 (Class 0), 0.91 (Class 1), 0.99 (Class 2)
- Confusion Matrix:
The model is highly accurate, there are occasional misclassifications, particularly between classes 0 and 1. Class 2 is predicted with the highest accuracy.

<ins>**Final Evaluation on Test Set:**</ins>
- Objective: Validate model performance on unseen data.
- Results:
   - Macro-F1 Score: 0.8631
   - Macro Precision: 0.8353
   - Macro Recall: 0.8975
   - Accuracy: 94%
- Confusion Matrix Summary: 
High performance across classes with strong generalization.
- Conclusion: 
The model shows robust performance and is suitable for real-world deployment.

<ins>**Model Performance Comparison:**</ins>
- Objective: Evaluate model performance improvements compared to baseline models.
   - Logistic Regression: Accuracy OF 86% , Macro F1-Score is  0.66
      - Challenges: Higher false positives, lower F1-score
   - Decision Tree:  Accuracy of 100% Macro F1-Score is  0.995
       - Strengths: High accuracy, but high training time and memory usage
    - Random Forest: Accuracy of 94% Macro F1-Score is  0.86
        - Advantages: Significant improvement over baselines, efficient training, and memory usage
- Summary: Random Forest shows significant performance gains with a balanced trade-off in efficiency compared to Logistic Regression and Decision Tree.

<ins>**Challenges faced:**</ins>
- Data Imbalance: Majority class (Benign Positive) dominating, causing skewed predictions.
- Model Overfitting: Initial models overfitted due to class imbalance and irrelevant features.
- High Dimensionality: Many features with potential noise, requiring careful feature selection.

<ins>**Solutions Implemented:**</ins>
- Data Imbalance: Techniques like SMOTE and class weighting.
- Model Overfitting: Cross-validation, regularization, and pruning.
- Feature Selection: Dimensionality reduction and importance analysis to identify key features.

<ins>**Future enhancements:**</ins>
- Continuous Learning: Implementing online learning algorithms to adapt to new data in real-time.
- Feature Engineering: Further exploration of new features, particularly those derived from domain knowledge.
- Model Optimization: Exploring advanced models like BERT for text analysis in incident descriptions.

<ins>**Recommendations:**</ins>
- Integration into SOC Workflows: Deploy the Random Forest model to automate triage and enhance response times.
- Data Collection: Increase data collection, especially for minority classes, to improve model robustness.
- Regular Monitoring: Continuous monitoring of model performance to adapt to evolving threats.

<ins>**Summary:**</ins>
The project successfully developed a machine learning model for SOCs, achieving high accuracy and efficiency in incident triage.

<ins>**Next Steps:**</ins>
Deployment, continuous improvement, and integration into broader cybersecurity strategies.


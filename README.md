# Titanic Survival Prediction Project

## Overview
The Titanic Survival Prediction project aims to predict the survival of passengers aboard the RMS Titanic using machine learning models. By leveraging the well-documented Titanic dataset, this project showcases data analysis, feature engineering, and classification techniques to achieve accurate predictions.

## Dataset
The dataset used for this project comes from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data). It consists of the following files:
- `train.csv`: Training data with survival labels.
- `test.csv`: Test data without survival labels (used for predictions).

### Key Features
- `PassengerId`: Unique identifier for each passenger.
- `Survived`: Survival indicator (0 = No, 1 = Yes).
- `Pclass`: Ticket class (1 = First, 2 = Second, 3 = Third).
- `Name`: Passenger name.
- `Sex`: Gender of the passenger.
- `Age`: Age of the passenger.
- `SibSp`: Number of siblings/spouses aboard.
- `Parch`: Number of parents/children aboard.
- `Ticket`: Ticket number.
- `Fare`: Ticket fare.
- `Cabin`: Cabin number.
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Steps Involved

### 1. Data Exploration
- Analyzed missing values and data distributions.
- Visualized key relationships between features and survival.

### 2. Data Preprocessing
- Filled missing values (e.g., median age for `Age`).
- Encoded categorical variables (`Sex`, `Embarked`).
- Standardized numerical features (e.g., `Fare`).

### 3. Feature Engineering
- Created new features such as:
  - Family size (`SibSp + Parch + 1`).
  - Title extraction from names (e.g., Mr., Mrs., Miss).
  - Fare bins and age groups.

### 4. Model Training
- Evaluated multiple machine learning models, including:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Support Vector Machines (SVM)
- Used grid search and cross-validation to fine-tune hyperparameters.

### 5. Model Evaluation
- Compared models using metrics like:
  - Accuracy
  - Precision, Recall, and F1-Score
  - ROC-AUC

### 6. Predictions
- Generated survival predictions for the test dataset.

## Results
- The best-performing model achieved an accuracy of approximately **XX%** on the validation set.
- Feature importance analysis revealed that `Sex`, `Pclass`, and `Age` were the most influential predictors of survival.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**:
  - Pandas and NumPy for data manipulation
  - Matplotlib and Seaborn for data visualization
  - Scikit-learn for machine learning

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd titanic-survival-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook titanic_survival.ipynb
   ```

## Directory Structure
```
.
├── data
│   ├── train.csv
│   ├── test.csv
├── notebooks
│   ├── titanic_survival.ipynb
├── models
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
├── requirements.txt
└── README.md
```

## Future Improvements
- Implement advanced feature selection techniques.
- Experiment with deep learning models.
- Incorporate ensemble methods for improved performance.

## Acknowledgments
- Kaggle for providing the Titanic dataset.
- Open-source libraries and resources used in this project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.


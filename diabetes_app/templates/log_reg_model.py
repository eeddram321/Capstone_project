import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np 

# Load dataset
df = pd.read_csv('diabetes.csv')

df['target'] = df['Outcome']
df.drop("Outcome", axis=1, inplace=True)

# Split into features and target
X = df.drop('target', axis=1)
y = df['target']

# Setup random seed
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Logistic Regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# GridSearchCV 
log_reg_grid = {"C": np.logspace(-4, 4, 30),
                "solver": ['liblinear']}

# Setup GridSearchCV for LogisticRegression
gs_log_reg = GridSearchCV(log_reg_model,
                         param_grid =log_reg_grid,
                         cv=5,
                         verbose=True)
# Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train);

print(gs_log_reg.score(X_test, y_test))

joblib.dump(log_reg_model, 'gs_log_reg_model.pkl')

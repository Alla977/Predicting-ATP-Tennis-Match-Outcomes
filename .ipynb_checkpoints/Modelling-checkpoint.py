#Loading data
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.inspection

X_train = pd.read_csv("Temp data/X_train.csv", sep=",")
X_test = pd.read_csv("Temp data/X_test.csv", sep=",")
y_train = pd.read_csv("Temp data/y_train.csv", sep=",")
y_test = pd.read_csv("Temp data/y_test.csv", sep=",")

#Label encoding categorical variables
# Initialize LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# Initialize LabelEncoder for categorical columns
categorical_columns = ['Main_player_name','surface', 'tourney_level','draw_size','Country','Home_crowd' ]  # Replace with the actual names of your categorical columns
label_encoders = {}
# Test set needs to contain Main_player_name from Train set in order to be able to make predictions on test set with Player name as variable
value_list=X_train['Main_player_name'].unique()
X_test = X_test[X_test['Main_player_name'].isin(value_list)]
y_test=y_test.loc[X_test.index]
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    X_train[col] = label_encoders[col].fit_transform(X_train[col])
    X_test[col] = label_encoders[col].transform(X_test[col])

# Fitting Random Forest Classifier (or Regressor for regression tasks)
from sklearn.ensemble import RandomForestClassifier
rf_model1 = RandomForestClassifier(n_estimators=300, random_state=42)  # You can adjust hyperparameters as needed
rf_model2 = RandomForestClassifier(n_estimators=300, random_state=42)
# Train the model on the training data
rf_model1.fit(X_train[['age_diff','height_diff','rank_diff','h_SVP_diff']], y_train)
rf_model2.fit(X_train, y_train)

y_pred1 = rf_model1.predict_proba(X_test[['age_diff','height_diff','rank_diff','h_SVP_diff']])[:,1]
y_pred2 = rf_model2.predict_proba(X_test)[:,1]

# Fitting a gradient boosting model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform="pandas")

gb_classifier1 = make_pipeline(scaler,GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, random_state=42))
gb_classifier1.fit(X_train[['age_diff','height_diff','rank_diff','h_SVP_diff','h_BPSP_diff_5']], y_train)
y_pred3 = gb_classifier1.predict_proba(X_test[['age_diff','height_diff','rank_diff','h_SVP_diff','h_BPSP_diff_5']])[:,1]


gb_classifier2 = make_pipeline(scaler,GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, random_state=42))
gb_classifier2.fit(X_train, y_train)
y_pred4 = gb_classifier2.predict_proba(X_test)[:,1]


# Define the GradientBoostingClassifier
gbm = GradientBoostingClassifier()

# Define hyperparameters and their possible values to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}

# Perform grid search
grid = GridSearchCV(estimator=gbm, param_grid=param_grid, scoring='accuracy', cv=3)
grid_result = grid.fit(X, y)

# Print the best hyperparameters and their corresponding accuracy
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")


## ROC
from sklearn.metrics import roc_curve, roc_auc_score, auc
fpr1, tpr1, thresholds = roc_curve(y_test, y_pred1)
fpr2, tpr2, thresholds = roc_curve(y_test, y_pred2)
fpr3, tpr3, thresholds = roc_curve(y_test, y_pred3)
fpr4, tpr4, thresholds = roc_curve(y_test, y_pred4)
fpr5, tpr5, thresholds = roc_curve(y_test, y_pred5)
fpr6, tpr6, thresholds = roc_curve(y_test, y_pred6)
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
roc_auc4 = auc(fpr4, tpr4)
roc_auc5 = auc(fpr5, tpr5)
roc_auc6 = auc(fpr6, tpr6)

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, color='darkorange', lw=.5, label=f'ROC for RF1 (AUC = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='darkblue', lw=.5, label=f'ROC for RF2 (AUC = {roc_auc2:.2f})')
plt.plot(fpr3, tpr3, color='red', lw=.5, label=f'ROC for GB1 (AUC = {roc_auc3:.2f})')
plt.plot(fpr4, tpr4, color='green', lw=.5, label=f'ROC for GB2 (AUC = {roc_auc4:.2f})')
plt.plot(fpr5, tpr5, color='black', lw=.5, label=f'ROC for Elastic net logistic regression (AUC = {roc_auc5:.2f})')
plt.plot(fpr6, tpr6, color='pink', lw=.5, label=f'ROC for Neural Network (AUC = {roc_auc6:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('ROC.png')
plt.show()






















## Fitting penalized logistic regression model
import numpy as np
from sklearn.preprocessing import OneHotEncoder

categorical_columns = ['Main_player_name', 'surface', 'tourney_level', 'draw_size', 'Country', 'Home_crowd']

onehot_encoders = {}
X_test_filtered = X_test.copy()  # Create a copy of the test data to apply one-hot encoding


for col in categorical_columns:
    encoder = OneHotEncoder()
    X_train_encoded = encoder.fit_transform(X_train[[col]])  # Fit and transform on the training data
    X_test_encoded = encoder.transform(X_test_filtered[[col]])  # Transform the test data using the same encoder
    onehot_encoders[col] = encoder

    # Update the X_train and X_test_filtered DataFrames with the one-hot encoded columns
    X_train = X_train.join(pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out([col])))
    X_test_filtered = X_test_filtered.join(pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out([col])))
    X_train=X_train.drop(categorical_columns, axis=1)
    X_test_filtered= X_test_filtered.drop(categorical_columns, axis=1)

# Assuming you have X_train and y_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the data

ratio = np.linspace(0, 1, 5)

# Fit the LogisticRegressionCV model
lr1 = LogisticRegressionCV(
    penalty='elasticnet',
    cv=5,
    random_state=43,
    solver='saga',
    l1_ratios=ratio
)
lr1.fit(X_train_scaled, y_train)

# Access coefficients from the best-fitted logistic regression model
coefficients = lr1.coef_

# If you want to transform coefficients to a Pandas DataFrame
coefficients_df = pd.DataFrame(coefficients, columns=X_train.columns)


## Fitting neural network models
import tensorflow as tf
from tensorflow import keras
import numpy as np
import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier


input_dim=np.shape(X_train)[1]
# Define a function to create the neural network model
def create_model(optimizer, activation, neurons):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create a KerasClassifier based on the create_model function
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameters and their possible values to search
param_grid = {
    'neurons': [8, 16, 32],
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'epochs': [10, 20],
    'batch_size': [32, 64]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_result = grid.fit(X_train, y_train)

# Print the best hyperparameters and their corresponding accuracy
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")y_pred6 = model.predict(X_test)



## ROC
from sklearn.metrics import roc_curve, roc_auc_score, auc
fpr1, tpr1, thresholds = roc_curve(y_test, y_pred1)
fpr2, tpr2, thresholds = roc_curve(y_test, y_pred2)
fpr3, tpr3, thresholds = roc_curve(y_test, y_pred3)
fpr4, tpr4, thresholds = roc_curve(y_test, y_pred4)
fpr5, tpr5, thresholds = roc_curve(y_test, y_pred5)
fpr6, tpr6, thresholds = roc_curve(y_test, y_pred6)
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
roc_auc4 = auc(fpr4, tpr4)
roc_auc5 = auc(fpr5, tpr5)
roc_auc6 = auc(fpr6, tpr6)

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, color='darkorange', lw=.5, label=f'ROC for RF1 (AUC = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='darkblue', lw=.5, label=f'ROC for RF2 (AUC = {roc_auc2:.2f})')
plt.plot(fpr3, tpr3, color='red', lw=.5, label=f'ROC for GB1 (AUC = {roc_auc3:.2f})')
plt.plot(fpr4, tpr4, color='green', lw=.5, label=f'ROC for GB2 (AUC = {roc_auc4:.2f})')
plt.plot(fpr5, tpr5, color='black', lw=.5, label=f'ROC for Elastic net logistic regression (AUC = {roc_auc5:.2f})')
plt.plot(fpr6, tpr6, color='pink', lw=.5, label=f'ROC for Neural Network (AUC = {roc_auc6:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('ROC.png')
plt.show()











## Hosmer-Lemeshow variant
import pandas as pd
import numpy as np
import seaborn as sns
#y_test = pd.read_csv("Temp data/y_test.csv", sep=",")
y_test=y_test.values
y_test=y_test.reshape(-1)
data = {
    'y_test': y_test,
    'y_pred1': y_pred1,
    'y_pred2': y_pred2,
    'y_pred3': y_pred3,
    'y_pred4': y_pred4,
    'y_pred5': y_pred5
}

df = pd.DataFrame(data)
melted_df = pd.melt(df,id_vars="y_test",var_name="Model_predictions",value_name="y_pred")
melted_df['y_pred_group']=pd.cut(melted_df['y_pred'], bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                                        right=False)
result = melted_df.groupby(['y_pred_group','Model_predictions'])[['y_test','y_pred']].mean().reset_index()
result_grouped = result.groupby('Model_predictions')

num_cols = min(len(result_grouped), 6)  # Ensure a maximum of 4 plots

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 10))

for i, (model, data_frame) in enumerate(result_grouped):
    # Convert 'y_pred_group' to its midpoint value
    data_frame['y_pred_group'] = data_frame['y_pred_group'].apply(lambda x: x.mid)

    # Calculate the row and column indices for the subplot
    row = i // 3
    col = i % 3

    # Use the current subplot
    ax = axes[row, col]

    ax.scatter(data_frame['y_pred_group'], data_frame['y_test'], label='y_test', marker='o')
    ax.scatter(data_frame['y_pred_group'], data_frame['y_pred'], label='y_pred', marker='x')

    ax.set_xlabel('y_pred_group')
    ax.set_ylabel('Values')
    ax.set_title(f'Plot for Model: {model}')
    ax.legend()
# Remove any empty subplots
for i in range(len(result_grouped), 12):
    row = i // 3
    col = i % 3
    fig.delaxes(axes[row, col])
plt.tight_layout()  # Adjust the layout to prevent overlap
plt.savefig('HL.png')
plt.show()


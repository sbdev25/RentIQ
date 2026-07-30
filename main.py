import pandas as pd
import numpy as np 
#for spliting data 
from sklearn.model_selection import train_test_split

#for evaluating the model 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score

#the model 

from sklearn.linear_model import LinearRegression

#for scaling 
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv('clean_data.csv')

# removes Nan
df_clean = df.dropna()

#test it after 
print(f"Lignes avant : {df.shape[0]} | Lignes après : {df_clean.shape[0]}")


#choose features and targets 
X_df = df_clean[['surface_m2', 'pieces', 'bedrooms', 'bathrooms' ]]
y_df = df_clean['price']

# convert it to numpy arrays 
X = X_df.to_numpy()
y = y_df.to_numpy()

# print("Features array shape (X):", X.shape)

# print(X_df.head(5))
# print("Labels array shape (y):", y.shape)
# print(y_df.head(5))

# 80% training 20% test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Verify the shapes
# print("Train set size (X_train):", X_train.shape)
# print("Test set size (X_test):", X_test.shape)
# the function responsible for scaling 
scaler = StandardScaler()

X_train_Scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_Scaled[:5])


model = LinearRegression()

model.fit(X_train_Scaled , y_train)

predictions = model.predict(X_test_scaled)


mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print(f"MSE = {mse}" )

print(f"RMSE = {rmse}" )

print(r2)


# ----------------------------

import matplotlib.pyplot as plt

# ---- 1. Predicted vs Actual ----
# The single most important diagnostic plot for regression.
# Perfect predictions would fall exactly on the diagonal line.
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual Price")
plt.legend()
plt.show()




import joblib

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved model.pkl and scaler.pkl")
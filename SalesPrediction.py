                         # sales_prediction.py #
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Create Synthetic Dataset (replace this with your CSV later)
data = {
    'ad_spend': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000],
    'platform': ['TV','Radio','Online','TV','Online','Radio','TV','Online','Radio','TV'],
    'segment': ['Youth','Adult','Youth','Adult','Senior','Youth','Adult','Senior','Youth','Adult'],
    'sales': [12000, 15000, 20000, 25000, 30000, 27000, 35000, 40000, 42000, 50000]
}

df = pd.DataFrame(data)

# Step 2: Split Data
X = df.drop(columns=['sales'])
y = df['sales']

# Step 3: Preprocessing (fixed with handle_unknown='ignore')
numeric_features = ['ad_spend']
categorical_features = ['platform','segment']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)   # FIXED
])

# Step 4: Train Models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for name, model in models.items():
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"---- {name} Results ----")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

# Step 5: Visualization (Advertising Impact)
plt.scatter(df['ad_spend'], df['sales'], color='blue')
plt.plot(df['ad_spend'], df['sales'], 'r--')
plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.title("Impact of Advertising Spend on Sales")
plt.show()

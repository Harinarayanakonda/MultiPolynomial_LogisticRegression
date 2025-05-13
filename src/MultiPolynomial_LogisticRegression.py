import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 1: Split into training and testing
# First reduce training set to make accuracy lower
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, test_size=0.5, random_state=42)

# Step 2: Add small noise to reduce accuracy
noise = np.random.normal(0, 0.1, X_train.shape)
X_train = X_train + noise

# Step 3: Polynomial Features with lower degree
poly = PolynomialFeatures(degree=1, include_bias=False)  # Degree 1 = Linear only
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Step 5: Train Logistic Regression model (no deprecation warning)
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 7: Prediction function for user input
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_poly = poly.transform(input_data)
    input_scaled = scaler.transform(input_poly)
    pred = model.predict(input_scaled)[0]
    return f"Predicted Iris Species: {iris.target_names[pred]}"

# Step 8: Example prediction (user input)
print(predict_species(6.3, 3.3 ,6.0, 2.5)) # Should predict 'setosa' or near it

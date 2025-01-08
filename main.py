import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import joblib

# 1. Load the dataset
data = pd.read_csv('parkinsons.csv')

# 2. Select features and target
# Replace 'PPE' and 'RPDE' with the actual column names for features
# Replace 'status' with the target column
X = data[['PPE', 'RPDE']]  # Two input features
y = data['status']  # Target output

# 3. Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train the model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

# 7. Save the model
joblib.dump(model, 'my_model.joblib')
print("Model saved as my_model.joblib")


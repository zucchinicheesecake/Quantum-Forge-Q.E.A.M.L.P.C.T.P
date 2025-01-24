# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define a function to train and integrate ML models
def train_ml_model(data):
    X_train, X_test, y_train, y_test = train_test_split(data['features'], data['target'], test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

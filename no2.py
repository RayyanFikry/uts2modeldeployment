import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

class LoanPredictionModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = RandomForestClassifier(random_state=42, n_estimators=100)

    def preprocess_data(self):
        X = self.data.drop(columns=['loan_status'])
        y = self.data['loan_status']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Random Forest Accuracy: {accuracy}")
        dump(self.model, 'best_rf_model.pkl', compress=3)

# Example usage
data_path = 'Dataset_A_loan.csv'
model = LoanPredictionModel(data_path)
model.preprocess_data()
model.train_model()
model.evaluate_model()

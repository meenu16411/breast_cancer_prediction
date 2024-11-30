# -*- coding: utf-8 -*-

# Breast Cancer Prediction Model Training

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Sklearn Imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

class BreastCancerModelTrainer:
    def __init__(self):
        # Initialize dataset attributes
        self.X = None
        self.y = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        self.selector = None

    def load_data(self):
        """
        Load Breast Cancer Dataset
        """
        breast_cancer = load_breast_cancer()
        self.X = breast_cancer.data
        self.y = breast_cancer.target
        self.feature_names = breast_cancer.feature_names
        return self.X, self.y

    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Data Preprocessing: Splitting and Scaling
        """
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        return self.X_train_scaled, self.X_test_scaled

    def feature_selection(self, k_best=10):
        """
        Perform Feature Selection
        """
        self.selector = SelectKBest(score_func=f_classif, k=k_best)
        X_train_selected = self.selector.fit_transform(self.X_train_scaled, self.y_train)

        # Get selected feature names
        selected_indices = self.selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]

        print("Selected Features:", selected_features)

        return X_train_selected

    def train_model(self):
        """
        Train ANN Model with Grid Search
        """
        # Define parameter grid
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }

        # Create MLPClassifier
        mlp = MLPClassifier(max_iter=1000)

        # Grid Search
        grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train_scaled, self.y_train)

        # Best model
        self.model = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)

        return self.model

    def evaluate_model(self):
        """
        Model Evaluation with Visualizations
        """
        # Predictions
        y_pred = self.model.predict(self.X_test_scaled)

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # Confusion Matrix Visualization
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()

        # ROC Curve
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def save_model_and_scaler(self, model_path='breast_cancer_model.pkl',
                               scaler_path='breast_cancer_scaler.pkl'):
        """
        Save trained model and scaler
        """
        # Save model
        joblib.dump(self.model, model_path)

        # Save scaler
        joblib.dump(self.scaler, scaler_path)

        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

def main():
    # Create model trainer instance
    trainer = BreastCancerModelTrainer()

    # Load data
    X, y = trainer.load_data()

    # Preprocess data
    X_train_scaled, X_test_scaled = trainer.preprocess_data()

    # Feature selection
    X_train_selected = trainer.feature_selection()

    # Train model
    model = trainer.train_model()

    # Evaluate model
    trainer.evaluate_model()

    # Save model and scaler
    trainer.save_model_and_scaler()

if __name__ == "__main__":
    main()


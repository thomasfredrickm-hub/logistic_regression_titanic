import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class LogisticRegression:
    """
    Logistic Regression implementation from scratch using NumPy.
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, 
                 regularization: float = 0.01, verbose: bool = True):
        """
        Initialize the Logistic Regression model.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        iterations : int
            Number of training iterations
        regularization : float
            L2 regularization parameter (lambda)
        verbose : bool
            Whether to print training progress
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.verbose = verbose
        self.weights = None
        self.bias = 0
        self.loss_history = []
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid activation function.
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss with L2 regularization.
        """
        m = len(y)
        epsilon = 1e-15  # To avoid log(0)
        
        # Binary cross-entropy
        loss = -np.mean(y * np.log(y_pred + epsilon) + 
                       (1 - y) * np.log(1 - y_pred + epsilon))
        
        # Add L2 regularization term
        reg_term = (self.regularization / (2 * m)) * np.sum(self.weights ** 2)
        
        return loss + reg_term
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features of shape (m, n)
        y : np.ndarray
            Training labels of shape (m,)
        """
        m, n = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0
        self.loss_history = []
        
        # Gradient descent
        for i in range(self.iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Backward pass - compute gradients
            dw = (1/m) * np.dot(X.T, (y_pred - y)) + (self.regularization/m) * self.weights
            db = (1/m) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}/{self.iterations}, Loss: {loss:.4f}")
        
        if self.verbose:
            print(f"Training completed! Final Loss: {self.loss_history[-1]:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for input samples.
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for input samples.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on absolute weight values.
        """
        return np.abs(self.weights)


def load_and_preprocess_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the Titanic dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable
    """
    # Load data
    df = pd.read_csv(filepath)
    
    print("="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nSurvival rate: {df['Survived'].mean():.2%}")
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 
                                        'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                        'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Create age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                            labels=[0, 1, 2, 3, 4])
    
    # Create fare groups
    df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Title encoding
    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    df['Title'] = df['Title'].map(title_mapping)
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                'Embarked', 'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']
    
    X = df[features].copy()
    y = df['Survived'].copy()
    
    print(f"\nFeatures selected: {features}")
    print(f"Feature matrix shape: {X.shape}")
    
    return X, y


def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize features using z-score standardization.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std
    
    return X_train_normalized, X_test_normalized


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    """
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    }


def plot_results(model: LogisticRegression, metrics: Dict, 
                X_train: pd.DataFrame, y_train: np.ndarray, 
                y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray):
    """
    Create comprehensive visualizations of the results.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Loss curve
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(model.loss_history, color='#3b82f6', linewidth=2)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title('Training Loss Curve', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix
    ax2 = plt.subplot(3, 3, 2)
    cm = np.array([[metrics['confusion_matrix']['TN'], metrics['confusion_matrix']['FP']],
                   [metrics['confusion_matrix']['FN'], metrics['confusion_matrix']['TP']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Died (0)', 'Survived (1)'],
                yticklabels=['Died (0)', 'Survived (1)'])
    plt.ylabel('Actual', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # 3. Feature Importance
    ax3 = plt.subplot(3, 3, 3)
    feature_names = X_train.columns
    importance = model.get_feature_importance()
    indices = np.argsort(importance)[::-1][:10]
    
    plt.barh(range(len(indices)), importance[indices], color='#8b5cf6')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Absolute Weight', fontsize=10)
    plt.title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # 4. Metrics Bar Chart
    ax4 = plt.subplot(3, 3, 4)
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                    metrics['f1_score'], metrics['specificity']]
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444']
    
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('Score', fontsize=10)
    plt.title('Model Performance Metrics', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Prediction Distribution
    ax5 = plt.subplot(3, 3, 5)
    survived_proba = y_pred_proba[y_test == 1]
    died_proba = y_pred_proba[y_test == 0]
    
    plt.hist(died_proba, bins=30, alpha=0.6, label='Died (Actual)', color='#ef4444')
    plt.hist(survived_proba, bins=30, alpha=0.6, label='Survived (Actual)', color='#10b981')
    plt.xlabel('Predicted Probability', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Threshold')
    
    # 6. ROC-like visualization (Probability threshold analysis)
    ax6 = plt.subplot(3, 3, 6)
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    
    for thresh in thresholds:
        preds = (y_pred_proba >= thresh).astype(int)
        acc = np.mean(preds == y_test)
        accuracies.append(acc)
    
    plt.plot(thresholds, accuracies, color='#3b82f6', linewidth=2)
    plt.xlabel('Threshold', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title('Accuracy vs Threshold', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # 7. Survival by Sex
    ax7 = plt.subplot(3, 3, 7)
    survival_by_sex = pd.DataFrame({'Sex': X_train.iloc[:, 1], 'Survived': y_train})
    survival_rate = survival_by_sex.groupby('Sex')['Survived'].mean()
    
    plt.bar(['Female', 'Male'], survival_rate.values, color=['#ec4899', '#3b82f6'], alpha=0.7)
    plt.ylabel('Survival Rate', fontsize=10)
    plt.title('Survival Rate by Sex', fontsize=12, fontweight='bold')
    plt.ylim(0, 1)
    
    for i, v in enumerate(survival_rate.values):
        plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # 8. Survival by Class
    ax8 = plt.subplot(3, 3, 8)
    survival_by_class = pd.DataFrame({'Pclass': X_train.iloc[:, 0], 'Survived': y_train})
    survival_rate = survival_by_class.groupby('Pclass')['Survived'].mean()
    
    plt.bar(['1st', '2nd', '3rd'], survival_rate.values, 
            color=['#fbbf24', '#a78bfa', '#f87171'], alpha=0.7)
    plt.ylabel('Survival Rate', fontsize=10)
    plt.title('Survival Rate by Class', fontsize=12, fontweight='bold')
    plt.ylim(0, 1)
    
    for i, v in enumerate(survival_rate.values):
        plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # 9. Age Distribution
    ax9 = plt.subplot(3, 3, 9)
    age_survived = X_train.iloc[y_train == 1, 2]
    age_died = X_train.iloc[y_train == 0, 2]
    
    plt.hist(age_died, bins=20, alpha=0.6, label='Died', color='#ef4444')
    plt.hist(age_survived, bins=20, alpha=0.6, label='Survived', color='#10b981')
    plt.xlabel('Age', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title('Age Distribution by Survival', fontsize=12, fontweight='bold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('titanic_logistic_regression_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'titanic_logistic_regression_results.png'")
    plt.show()


def main():
    """
    Main function to run the complete pipeline.
    """
    print("\n" + "="*60)
    print("TITANIC SURVIVAL PREDICTION - LOGISTIC REGRESSION")
    print("="*60 + "\n")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data('titanic.csv')
    
    # Split data (80-20 split)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Normalize features
    X_train_norm, X_test_norm = normalize_features(X_train.values, X_test.values)
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    model = LogisticRegression(learning_rate=0.1, iterations=1000, 
                              regularization=0.01, verbose=True)
    model.fit(X_train_norm, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_norm)
    y_pred_proba = model.predict_proba(X_test_norm)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    
    print("\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {metrics['confusion_matrix']['TN']}")
    print(f"  False Positives (FP): {metrics['confusion_matrix']['FP']}")
    print(f"  False Negatives (FN): {metrics['confusion_matrix']['FN']}")
    print(f"  True Positives (TP):  {metrics['confusion_matrix']['TP']}")
    
    # Sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (First 10 test cases)")
    print("="*60)
    print(f"{'#':<4} {'Actual':<8} {'Predicted':<10} {'Probability':<12} {'Result':<8}")
    print("-" * 60)
    
    for i in range(min(10, len(y_test))):
        result = "✓" if y_test[i] == y_pred[i] else "✗"
        print(f"{i+1:<4} {y_test[i]:<8} {y_pred[i]:<10} {y_pred_proba[i]:.4f} ({y_pred_proba[i]*100:.1f}%){'':<2} {result:<8}")
    
    # Plot results
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    plot_results(model, metrics, X_train, y_train, y_test, y_pred, y_pred_proba)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
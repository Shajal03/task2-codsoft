import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 10000

# Generate synthetic features similar to V1-V28 using a normal distribution
V_features = np.random.randn(n_samples, 28)

# Generate synthetic 'Amount' feature using an exponential distribution
amount = np.random.exponential(scale=100, size=n_samples)

# Generate synthetic 'Class' feature with 0 for legitimate and 1 for fraudulent
# Assume 1% of the transactions are fraudulent
fraud_percentage = 0.01
classes = np.random.choice([0, 1], size=n_samples, p=[1-fraud_percentage, fraud_percentage])

# Combine all features into a DataFrame
columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
data = np.hstack((V_features, amount.reshape(-1, 1), classes.reshape(-1, 1)))
df = pd.DataFrame(data, columns=columns)

# Display the first few rows of the synthetic dataset
print(df.head())

# Save the synthetic dataset to a CSV file
df.to_csv('creditcard.csv', index=False)

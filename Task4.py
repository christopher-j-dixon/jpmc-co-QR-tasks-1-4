import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data file
df = pd.read_csv('Loan_Data.csv')

# Inspect & check the datafile
# NOTE: We have already done this in task 3

# Inspcet the FICO Score feature
print(df['fico_score'].describe())

# Plot the FICO Score feature histogram
plt.figure(figsize=(12, 8))
sns.histplot(df['fico_score'], bins=50, kde=True)
plt.title('FICO Score Histogram')
plt.xlabel('FICO score')
plt.ylabel('Number of Customers')
plt.grid(True)

def log_likelihood_buckets(data, n_buckets, method='quantile'):
    """
    Calculate the log-likelihood for given bucketing method.
    
    Parameters:
    - data: DataFrame containing FICO scores and default information
    - n_buckets: Number of buckets
    - method: Method for bucketing ('quantile' or 'equal_width')
    
    Returns:
    - Log-likelihood value for the given bucket configuration
    """
    
    # Determine boundaries based on the method
    if method == 'quantile':
        quantiles = np.linspace(1/n_buckets, 1-1/n_buckets, n_buckets-1)
        boundaries = [300] + list(data['fico_score'].quantile(quantiles)) + [850]
    elif method == 'equal_width':
        boundaries = n_buckets
    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'quantile' or 'equal_width'.")
    
    # Create bins using the determined boundaries
    bins = pd.cut(data['fico_score'], bins=boundaries, include_lowest=True)
    
    # Calculate the number of records and defaults in each bin
    counts = bins.value_counts().sort_index()
    defaults = data.groupby(bins)['default'].sum()
    
    # Calculate the probability of default in each bucket
    probs = defaults / counts
    
    # Calculate the log-likelihood
    ll = np.sum(defaults * np.log(probs) + (counts - defaults) * np.log(1 - probs))
    
    return ll

# Define a range for the number of buckets
bucket_range = range(2, 21)

# Compute log-likelihood values for each number of buckets
ll_values_equal_width = [log_likelihood_buckets(df, n, 'equal_width') for n in bucket_range]
ll_values_quantile = [log_likelihood_buckets(df, n, 'quantile') for n in bucket_range]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(bucket_range, ll_values_equal_width, marker='o', label="Equal Width Method")
plt.plot(bucket_range, ll_values_quantile, marker='o', label="Quantile Method")
plt.xlabel('Number of Buckets')
plt.ylabel('Log-Likelihood')
plt.title('Comparison of Log-Likelihood Values: Equal Width vs. Quantile Bucketing')
plt.grid(True)
plt.legend()  # Add legend to the plot
plt.tight_layout()
plt.show()

#  From the above modelling, I am going to chose n = 6  

# Predict Default Rate function
def predict_default_rate_corrected(fico_score, data):
    """
    Predict the default rate based on the given FICO score.
    
    Parameters:
    - fico_score: The FICO score of the borrower.
    - data: DataFrame containing FICO scores and default information
    
    Returns:
    - A message detailing the rating, default rate percentile, and associated comment.
    """
    
    # Determine boundaries based on the data
    n_buckets = 6  # As previously decided
    quantiles = np.linspace(1/n_buckets, 1-1/n_buckets, n_buckets-1)
    boundaries = [300] + list(data['fico_score'].quantile(quantiles)) + [850]
    
    # Determine the rating and associated interval based on the FICO score and boundaries
    rating_interval = pd.cut([fico_score], bins=boundaries, include_lowest=True)[0]
    rating = pd.cut([fico_score], bins=boundaries, labels=range(6, 0, -1), include_lowest=True)[0]
    
    # Calculate default rates for each rating
    bins = pd.cut(data['fico_score'], bins=boundaries, include_lowest=True)
    defaults = data.groupby(bins)['default'].sum()
    counts = bins.value_counts().sort_index()
    default_rates = defaults / counts
    
    # Get the default rate for the determined rating using the interval
    rate = default_rates.loc[rating_interval]
    
    # Comment based on the rating
    comments = {
        1: "Very low chance of default",
        2: "Low chance of default",
        3: "Below average chance of default",
        4: "Above average chance of default",
        5: "High chance of default",
        6: "Very high chance of default"
    }
    comment = comments[int(rating)]
    
    return comment



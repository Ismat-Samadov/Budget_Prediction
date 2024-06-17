from scipy.stats import median_abs_deviation
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'budget.xlsx'
budget_data = pd.read_excel(file_path)

# Calculate initial average expense
initial_average_expense = budget_data['amount'].mean()

# Method 1: Z-Score
z_scores = stats.zscore(budget_data['amount'])
abs_z_scores = abs(z_scores)
filtered_entries_z = (abs_z_scores < 3)
data_z = budget_data[filtered_entries_z]

# Method 2: IQR
Q1 = budget_data['amount'].quantile(0.25)
Q3 = budget_data['amount'].quantile(0.75)
IQR = Q3 - Q1
filtered_entries_iqr = (budget_data['amount'] >= (Q1 - 1.5 * IQR)) & (budget_data['amount'] <= (Q3 + 1.5 * IQR))
data_iqr = budget_data[filtered_entries_iqr]

# Method 3: Modified Z-Score
median_amount = budget_data['amount'].median()
mad = median_abs_deviation(budget_data['amount'])
modified_z_scores = 0.6745 * (budget_data['amount'] - median_amount) / mad
abs_modified_z_scores = abs(modified_z_scores)
filtered_entries_modified_z = (abs_modified_z_scores < 3.5)
data_modified_z = budget_data[filtered_entries_modified_z]

# Method 4: Percentile-based
lower_bound = budget_data['amount'].quantile(0.01)
upper_bound = budget_data['amount'].quantile(0.99)
filtered_entries_percentile = (budget_data['amount'] >= lower_bound) & (budget_data['amount'] <= upper_bound)
data_percentile = budget_data[filtered_entries_percentile]

# Method 5: Hampel Filter
median_absolute_deviation = mad
filtered_entries_hampel = abs(budget_data['amount'] - median_amount) <= (3 * median_absolute_deviation)
data_hampel = budget_data[filtered_entries_hampel]

# Calculate average expenses after removing outliers
average_expense_z = data_z['amount'].mean()
average_expense_iqr = data_iqr['amount'].mean()
average_expense_modified_z = data_modified_z['amount'].mean()
average_expense_percentile = data_percentile['amount'].mean()
average_expense_hampel = data_hampel['amount'].mean()

# Create a dataframe to summarize the results
results = pd.DataFrame({
    'Method': ['Initial', 'Z-Score', 'IQR', 'Modified Z-Score', 'Percentile', 'Hampel'],
    'Average Expense (AZN)': [
        initial_average_expense, 
        average_expense_z, 
        average_expense_iqr, 
        average_expense_modified_z, 
        average_expense_percentile, 
        average_expense_hampel
    ]
})

# Plot the results
plt.figure(figsize=(12, 6))
plt.bar(results['Method'], results['Average Expense (AZN)'], color='mediumseagreen')
plt.xlabel('Method')
plt.ylabel('Average Expense (AZN)')
plt.title('Average Expense Before and After Outlier Removal')
plt.grid(True)
plt.tight_layout()
plt.show()

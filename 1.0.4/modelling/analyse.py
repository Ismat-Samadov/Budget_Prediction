import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the Excel file
file_path = 'budget.xlsx'
xls = pd.ExcelFile(file_path)

# Display sheet names to understand the structure of the file
sheet_names = xls.sheet_names
sheet_names

df = pd.read_excel(file_path, sheet_name='budjet')
# Convert datetime column to pandas datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Time range of the data
time_range = df['datetime'].min(), df['datetime'].max()

# Category breakdown
category_breakdown = df.groupby('category')['amount'].sum().reset_index()

# Monthly spending
df['month'] = df['datetime'].dt.to_period('M')
monthly_spending = df.groupby('month')['amount'].sum().reset_index()

# Top categories by spending
top_categories = category_breakdown.sort_values(by='amount', ascending=False).head()



# Monthly spending trend visualization
plt.figure(figsize=(12, 6))
plt.plot(monthly_spending['month'].astype(str), monthly_spending['amount'], marker='o')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Amount (AZN)')
plt.title('Monthly Spending Trend')
plt.grid(True)
plt.tight_layout()
plt.show()

# Category breakdown visualization
plt.figure(figsize=(12, 6))
plt.barh(category_breakdown['category'], category_breakdown['amount'], color='skyblue')
plt.xlabel('Amount (AZN)')
plt.ylabel('Category')
plt.title('Total Spending by Category')
plt.grid(True)
plt.tight_layout()
plt.show()

# Top categories visualization
plt.figure(figsize=(12, 6))
plt.bar(top_categories['category'], top_categories['amount'], color='salmon')
plt.xlabel('Category')
plt.ylabel('Amount (AZN)')
plt.title('Top 5 Spending Categories')
plt.grid(True)
plt.tight_layout()
plt.show()

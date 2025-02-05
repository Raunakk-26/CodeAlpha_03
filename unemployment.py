import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
file_path = "C:/Users/rauna/OneDrive/Desktop/COde/Unemployment.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
print("Cleaned Column Names:", df.columns)
print("Missing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
df = df.drop_duplicates()
df['Estimated Unemployment Rate (%)'] = df['Estimated Unemployment Rate (%)'].fillna(df['Estimated Unemployment Rate (%)'].median())
df['Estimated Employed'] = df['Estimated Employed'].fillna(df['Estimated Employed'].median())
df['Estimated Labour Participation Rate (%)'] = df['Estimated Labour Participation Rate (%)'].fillna(df['Estimated Labour Participation Rate (%)'].median())
print("\nBasic Statistics Summary:")
print(df.describe())
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_cols]
plt.figure(figsize=(8, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=df)  
plt.title('Unemployment Rate in India Over Time')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.show()
plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=df)  
plt.xticks(rotation=90)
plt.title('Unemployment Rate by Region')
plt.show()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  
df.set_index('Date', inplace=True)
plt.figure(figsize=(12, 6))
df['Estimated Unemployment Rate (%)'].plot()
plt.title('Time Series of Unemployment Rate')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.show()
X = df[['Estimated Employed', 'Estimated Labour Participation Rate (%)']]  
y = df['Estimated Unemployment Rate (%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error: {mse}')
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Unemployment Rate')
plt.ylabel('Predicted Unemployment Rate')
plt.title('Actual vs Predicted Unemployment Rate')
plt.show()

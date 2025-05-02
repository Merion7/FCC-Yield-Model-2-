import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  

# Load data  
try:  
    data = pd.read_csv('fcc_data.csv')  
    data.columns = data.columns.str.strip()  
    print("Columns in CSV:", data.columns.tolist())  
except:  
    print("ERROR: Couldn't find 'fcc_data.csv'!")  
    exit()  

# Calculate stats  
mean_yield = data['Gasoline_Yield'].mean()  
max_yield = data['Gasoline_Yield'].max()  
min_yield = data['Gasoline_Yield'].min()  

print(f"📊 Average Yield: {mean_yield:.1f}%")  
print(f"🚀 Max Yield: {max_yield:.1f}%")  
print(f"📉 Min Yield: {min_yield:.1f}%")  

# Train prediction model  
X = data[['Temperature']]  
y = data['Gasoline_Yield']  
model = LinearRegression()  
model.fit(X, y)  
predictions = model.predict(X)  

# Plot  
plt.scatter(data['Temperature'], data['Gasoline_Yield'], label='Actual Yield')  
plt.plot(data['Temperature'], predictions, color='red', label='Predicted Yield')  
plt.xlabel('Temperature (°C)')  
plt.ylabel('Gasoline Yield (%)')  
plt.title('FCC Yield vs. Temperature (with Predictions)')  
plt.legend()  
plt.show()  
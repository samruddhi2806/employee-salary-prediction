import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Experience': [1, 3, 5, 7, 10],
    'Salary': [40000, 60000, 75000, 100000, 120000]
})

x = data[['Experience']]
y = data['Salary']
model = LinearRegression()
model.fit(x, y)

st.title("Salary Prediction App")
experience = st.slider("Years of Experience", 0, 20, 1)
predicted_salary = model.predict(np.array([[experience]]))[0]
st.success(f"Predicted Salary: ₹{predicted_salary:,.2f}")

plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.legend()
st.pyplot(plt)


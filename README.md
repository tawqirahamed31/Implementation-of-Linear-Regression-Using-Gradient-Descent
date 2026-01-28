# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ab = pd.read_csv(r"C:\Users\acer\Desktop\ml\ex3.csv")
x_raw = ab['R&D Spend'].values
y_raw = ab['Profit'].values

x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

w = 0.0
b = 0.0
alpha = 0.01
epochs = 900
n = len(x)

losses = []

for _ in range(epochs):
    y_hat = w * x + b

    # Mean Squared Error
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y, color="blue", label="Data")
plt.plot(x, w * x + b, color="orange", label="Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
```

## Output:
<img width="1247" height="630" alt="image" src="https://github.com/user-attachments/assets/30d34a3e-c05e-42ab-acd9-2d728ebae50b" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

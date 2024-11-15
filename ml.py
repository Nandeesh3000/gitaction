import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

height = [[3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0]]
weight = [6, 8, 10, 12, 14, 16, 18, 20, 22]

plt.scatter([h[0] for h in height], weight, color='black')
plt.xlabel("Height")
plt.ylabel("Weight")

reg = linear_model.LinearRegression()
reg.fit(height, weight)

X_height = [[182.0]]
predicted_weight = reg.predict(X_height)
print(f"Predicted weight for height {X_height[0][0]}: {predicted_weight[0]}")

plt.show()

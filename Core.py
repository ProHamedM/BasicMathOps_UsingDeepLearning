import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Generate Math Dataset (Inputs: Num1, Num2, Op_Code -> Output: Result)
def generate_math_data(num_samples=100000):
    X = []
    y = []
    for _ in range(num_samples):
        n1 = np.random.uniform(1, 100)
        n2 = np.random.uniform(1, 100)
        op = np.random.randint(0, 4) # 0:+, 1:-, 2:*, 3:/
        
        if op == 0: res = n1 + n2
        elif op == 1: res = n1 - n2
        elif op == 2: res = n1 * n2
        else: res = n1 / n2
        
        X.append([n1, n2, op])
        y.append(res)
    return np.array(X), np.array(y)

# 2. Build the Model
X_train, y_train = generate_math_data()

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(3,)), # Input: n1, n2, op
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1) # Output: The result of the math
])

model.compile(optimizer='adam', loss='mse')

# 3. Train the model to "learn" math
print("Training model to learn basic math functions...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# 4. Testing the Model
def test_math(n1, n2, op_name):
    ops = {'+':0, '-':1, '*':2, '/':3}
    prediction = model.predict([[n1, n2, ops[op_name]]], verbose=0)
    print(f"Question: {n1} {op_name} {n2} | Model Prediction: {prediction[0][0]:.2f}")

test_math(10, 5, '+')
test_math(20, 4, '/')
test_math(6, 7, '*')

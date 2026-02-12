import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

# 1. Generate Math Dataset with better structure
def generate_math_data(num_samples=100000):
    X = []
    y = []
    for _ in range(num_samples):
        n1 = np.random.uniform(1, 100)
        n2 = np.random.uniform(1, 100)
        op = np.random.randint(0, 4)  # 0:+, 1:-, 2:*, 3:/
        
        # Handle division by zero
        if op == 0: 
            res = n1 + n2
        elif op == 1: 
            res = n1 - n2
        elif op == 2: 
            res = n1 * n2
        else: 
            res = n1 / n2 if n2 != 0 else n1  # Avoid division by zero
        
        X.append([n1, n2, op])
        y.append(res)
    
    return np.array(X), np.array(y)

# 2. Generate data
print("Generating dataset...")
X_train, y_train = generate_math_data(num_samples=100000)

# 3. Normalize features (IMPORTANT: scales n1, n2, op properly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Split into train and validation sets
split_idx = int(0.8 * len(X_train_scaled))
X_train_split = X_train_scaled[:split_idx]
y_train_split = y_train[:split_idx]
X_val = X_train_scaled[split_idx:]
y_val = y_train[split_idx:]

# 5. Build improved model architecture
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(3,)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output: result
])

# 6. Compile with better settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 7. Train with validation
print("Training model to learn basic math functions...")
history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    verbose=1
)

# 8. Evaluate on test set
test_loss, test_mae = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Loss: {test_loss:.4f}")
print(f"Validation MAE: {test_mae:.4f}")

# 9. Testing the Model with scaled inputs
def test_math(n1, n2, op_name):
    ops = {'+': 0, '-': 1, '*': 2, '/': 3}
    
    # Handle division by zero
    if op_name == '/' and n2 == 0:
        print(f"Question: {n1} {op_name} {n2} | ERROR: Division by zero!")
        return
    
    # Scale input using the same scaler
    input_scaled = scaler.transform([[n1, n2, ops[op_name]]])
    prediction = model.predict(input_scaled, verbose=0)
    
    # Calculate actual result
    if op_name == '+': actual = n1 + n2
    elif op_name == '-': actual = n1 - n2
    elif op_name == '*': actual = n1 * n2
    else: actual = n1 / n2
    
    error = abs(prediction[0][0] - actual)
    print(f"Question: {n1} {op_name} {n2}")
    print(f"  Actual: {actual:.2f}")
    print(f"  Prediction: {prediction[0][0]:.2f}")
    print(f"  Error: {error:.4f}\n")

# Test cases
print("\n=== Testing Results ===")
test_math(10, 5, '+')
test_math(20, 4, '/')
test_math(6, 7, '*')
test_math(50, 10, '-')

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load data
train_data = np.load('training-val-test-data.npz')
u_train = train_data['u']
th_train = train_data['th']

test_data = np.load('hidden-test-simulation-submission-file.npz')
u_test = test_data['u']
th_test = test_data['th']  # only first 50 entries filled

# Parameters
na = 20  # number of past angles
nb = 20  # number of past inputs
timesteps = na + nb

# Helper function to prepare sequences
def create_lstm_sequences(u, th, na, nb):
    X = []
    Y = []
    for k in range(max(na, nb), len(th)):
        u_seq = u[k-nb:k].reshape(-1, 1)
        th_seq = th[k-na:k].reshape(-1, 1)
        x_seq = np.vstack([u_seq, th_seq])  # shape: (nb+na, 1)
        X.append(x_seq)
        Y.append(th[k])
    return np.array(X), np.array(Y)

# Prepare training data
X_train, Y_train = create_lstm_sequences(u_train, th_train, na, nb)

# Normalize data
u_mean, u_std = np.mean(u_train), np.std(u_train)
th_mean, th_std = np.mean(th_train), np.std(th_train)

X_train = (X_train - np.array([[u_mean]*nb + [th_mean]*na]).reshape(1, timesteps, 1)) / np.array([[u_std]*nb + [th_std]*na]).reshape(1, timesteps, 1)
Y_train = (Y_train - th_mean) / th_std

# Build LSTM model
model = Sequential([
    InputLayer(input_shape=(timesteps, 1)),
    LSTM(32, activation='sigmoid', return_sequences=True),
    #LSTM(16, activation='sigmoid', return_sequences=True),
    LSTM(16, activation='sigmoid'),
    Dense(1),
    Dense(1),
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train model and store history
early_stop = EarlyStopping(patience=20, restore_best_weights=True, verbose=1)
history = model.fit(X_train, Y_train, epochs=500, batch_size=32,
                    validation_split=0.1, callbacks=[early_stop], verbose=1)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Simulation function
def simulation_lstm_model(model, ulist, ylist, na, nb, skip=50, normalize=True):
    assert skip >= max(na, nb), f"Skip={skip} must be >= max(na, nb)={max(na, nb)}"
    assert len(ulist) > skip, f"Input length {len(ulist)} must be > skip={skip}"
    
    Y = ylist[:skip].tolist()
    upast = ulist[skip-nb:skip].tolist()
    ypast = ylist[skip-na:skip].tolist()
    
    print("Starting simulation...")
    print(f"Initial upast (len={len(upast)}): {upast}")
    print(f"Initial ypast (len={len(ypast)}): {ypast}")
    print(f"Total input length: {len(ulist)}")
    
    for i, u in enumerate(ulist[skip:], start=skip):
        x_seq = np.array(upast + ypast).reshape(1, na+nb, 1)
        print(f"Step {i}: x_seq shape = {x_seq.shape}")
        
        if normalize:
            mean_vec = np.array([[u_mean]*nb + [th_mean]*na]).reshape(1, na+nb, 1)
            std_vec = np.array([[u_std]*nb + [th_std]*na]).reshape(1, na+nb, 1)
            # Guard against zero std
            std_vec = np.where(std_vec == 0, 1e-8, std_vec)
            x_seq = (x_seq - mean_vec) / std_vec
        
        y_pred = model.predict(x_seq, verbose=0)[0, 0]
        
        if np.isnan(y_pred):
            raise ValueError(f"Prediction produced NaN at step {i}!")
        
        if normalize:
            y_pred = y_pred * th_std + th_mean
        
        print(f"Step {i}: input u={u:.6f}, predicted y={y_pred:.6f}")
        
        Y.append(y_pred)
        
        upast.append(u)
        upast.pop(0)
        
        ypast.append(y_pred)
        ypast.pop(0)
    
    return np.array(Y)

u_mean = np.mean(u_train)
u_std = np.std(u_train) or 1e-8
th_mean = np.mean(th_train)
th_std = np.std(th_train) or 1e-8

print(f"u_mean={u_mean:.6f}, u_std={u_std:.6f}")
print(f"th_mean={th_mean:.6f}, th_std={th_std:.6f}")

# Evaluate on training data
# skip = max(na, nb)
# th_train_sim = simulation_lstm_model(model, u_train, th_train, na, nb, skip=skip)
# rms_train = np.mean((th_train_sim[skip:] - th_train[skip:]) ** 2) ** 0.5
# print(f"Train Simulation RMS Error: {rms_train:.6f} radians | {(rms_train / (2*np.pi) * 360):.3f} degrees")

# Simulate on test data
skip = 50
th_test_sim = simulation_lstm_model(model, u_test, th_test, na, nb, skip=skip)

# rms_train = np.mean((th_test[skip:] - th_train[skip:5000]) ** 2) ** 0.5
# print(f"Train Simulation RMS Error: {rms_train:.6f} radians | {(rms_train / (2*np.pi) * 360):.3f} degrees")

# Save output for submission
np.savez('hidden-test-simulation-lstm-submission-file.npz', th=th_test_sim, u=u_test)


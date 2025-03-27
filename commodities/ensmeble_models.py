import numpy as np
import matplotlib.pyplot as plt
import markovify

# Simulation Parameters
T = 1.0  # 1 year
N = 252  # Daily steps
dt = T / N
num_paths = 500  # Number of simulated paths

# Model Parameters
mu_y = 0.02  # Drift of long-term factor
sigma_y = 0.15  # Volatility of long-term factor
kappa_x = 2.0  # Mean-reversion speed of short-term factor
theta_x = 0.0  # Mean level of short-term factor
sigma_x = 0.2  # Volatility of short-term factor
lambda_j = 0.5  # Jump frequency (per year)
mu_j = -0.1  # Mean jump size
sigma_j = 0.2  # Jump volatility

# Generate synthetic log returns for training
log_returns = np.random.randn(N, 1) * 0.02  # Simulated return series

# Train a Markov model on the log return series using markovify
# Convert log returns into a text-like structure (discretize values)
discretized_returns = np.digitize(log_returns, bins=np.linspace(-0.1, 0.1, 20))  # Discretize log returns into states
return_series = ' '.join(map(str, discretized_returns.flatten()))  # Convert to string format for markovify

# Create a Markov model using markovify
text_model = markovify.Text(return_series, state_size=2)

# Initialize paths
S_markov = np.zeros((N, num_paths))
S_weighted = np.zeros((N, num_paths))
S_markov[0, :] = 100
S_weighted[0, :] = 100

# Simulating paths with Markov switching based on Markov model
for i in range(1, N):
    dW_y = np.random.randn(num_paths) * np.sqrt(dt)
    
    # Markov-based switching using Markov model from markovify
    regime = [int(text_model.make_sentence().split()[0]) for _ in range(num_paths)]  # Use the first word as the state
    
    # Jumps and volatility adjustment for each regime
    Jumps = np.random.poisson(lambda_j * dt, num_paths) * (mu_j + sigma_j * np.random.randn(num_paths))
    S_markov[i, :] = S_markov[i - 1, :] * np.exp((mu_y - 0.5 * sigma_y**2) * dt + sigma_y * dW_y + np.array(regime) * Jumps)
    
    # Weighted blending (fix for division by zero)
    std_devs = np.std(S_weighted[max(i-10,0):i, :], axis=0)
    max_std_dev = np.max(std_devs)

    # Avoid division by zero by checking if max_std_dev is non-zero
    if max_std_dev > 0:
        vol_weight = np.clip(std_devs / max_std_dev, 0, 1)
    else:
        vol_weight = np.ones(num_paths)  # If std is zero, use full weight
    normal_weight = 1 - vol_weight
    Jumps_weighted = np.random.poisson(lambda_j * dt, num_paths) * (mu_j + sigma_j * np.random.randn(num_paths))
    S_weighted[i, :] = S_weighted[i - 1, :] * np.exp((mu_y - 0.5 * sigma_y**2) * dt + sigma_y * dW_y + vol_weight * Jumps_weighted)

# Plot Sample Paths
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
for j in range(num_paths):
    plt.plot(S_markov[:, j], label=f'Path {j+1}')
plt.ylim([200,0])
plt.xlabel('Time Steps')
plt.ylabel('Commodity Price')
plt.title('Markov Chain Switching (Markovify)')

plt.subplot(2, 2, 2)
for j in range(num_paths):
    plt.plot(S_weighted[:, j], label=f'Path {j+1}')
plt.xlabel('Time Steps')
plt.ylabel('Commodity Price')
plt.title('Weighted Model Averaging')
# plt.show()

# Compute log returns
log_returns_markov = np.diff(np.log(S_markov), axis=0).flatten()
log_returns_markov = log_returns_markov[np.abs(log_returns_markov) < 0.05]
log_returns_weighted = np.diff(np.log(S_weighted), axis=0).flatten()
log_returns_weighted = log_returns_weighted[np.abs(log_returns_weighted) < 0.05]

# Plot log return distributions
# plt.figure(figsize=(12, 5))
plt.subplot(2, 2, 3)
plt.hist(log_returns_markov, bins=200, density=True, alpha=0.6, color='b')
plt.title('Log Return Distribution - Markov Chain Switching (Markovify)')
plt.xlim([-0.05,0.05])
plt.xlabel('Log Return')
plt.ylabel('Density')

plt.subplot(2, 2, 4)
plt.hist(log_returns_weighted, bins=200, density=True, alpha=0.6, color='r')
plt.title('Log Return Distribution - Weighted Averaging')
plt.xlim([-0.05,0.05])
plt.xlabel('Log Return')
plt.ylabel('Density')
plt.show()

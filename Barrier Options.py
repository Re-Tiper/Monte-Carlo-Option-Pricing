import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import scipy.stats


def PriceBS(S0, K, r, T, sigma, call_or_put= 'c'):
    N = scipy.stats.norm.cdf
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call_or_put == 'c':
        return N(d1) * S0 - N(d2) * K * np.exp(-r*T)
    elif call_or_put == 'p':
        return N(-d2) * K * np.exp(-r*T) - N(-d1) * S0
    else:
        return "Specify call or put options."

def GeneratePaths(mu, sigma, S0, M, n, T):
    # mu = drift, sigma = volatility, S0 = initial stock price, M = number of simulations, n = steps, T = time
    # Calculate each time step
    dt = T / n
    # Simulation using numpy arrays
    St = np.exp(
        (mu - 0.5 * sigma ** 2 ) * dt
        + sigma * np.sqrt(dt) * np.random.normal(0, 1, size=(n, M))
    )
    # Include array of 1's
    St = np.vstack([np.ones(M), St])  # vertically stack a row of ones in the array St
    # Multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
    St = S0 * St.cumprod(axis=0) # each column represents a path
    return St

def PlotPaths(paths, S0, M, n, T,  barrier_crossed= None, knockin=None, knockout=None):
    # Define time interval correctly
    time = np.linspace(0, T, n + 1)
    # Require numpy array that is the same shape as St
    tt = np.full(shape=(M, n + 1), fill_value=time).T

    if knockout: # Up,Down and Out
        # Plot paths that crossed the barrier in red
        plt.plot(tt[:, barrier_crossed], paths[:, barrier_crossed], 'r-', linewidth=0.8, alpha=0.5)
        # Plot paths that didn't cross the barrier in green
        plt.plot(tt[:, ~barrier_crossed], paths[:, ~barrier_crossed], 'g-', linewidth=0.8, alpha=0.5)
        # Plot barrier line
        plt.plot([0, T], [knockout, knockout], 'k-', linewidth=2.0)
    if knockin: # Up,Down and In
        # Plot paths that crossed the barrier in green
        plt.plot(tt[:, barrier_crossed], paths[:, barrier_crossed], 'g-', linewidth=0.8, alpha=0.5)
        # Plot paths that didn't cross the barrier in red
        plt.plot(tt[:, ~barrier_crossed], paths[:, ~barrier_crossed], 'r-', linewidth=0.8, alpha=0.5)
        # Plot barrier line
        plt.plot([0, T], [knockin, knockin], 'k-', linewidth=2.0)
    elif knockin is None and knockout is None:
        plt.plot(tt, paths, linewidth=0.8)

    plt.xlabel("Time $(t)$", fontsize=16)
    plt.ylabel("Value $(S_t)$", fontsize=16)
    plt.title(
        "$dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(
            S0, mu, sigma), fontsize=22)
    plt.show()


def PriceOption(mu, sigma, S0, M, n, T, K, call_or_put='c', knockin=None, knockout=None):
    # K= strike, knockin= barrier level for knockin option, knockout= barrier level for knockout option
    start_time = time.time()
    if knockin and knockout:
        raise Exception("Can't have 2 barriers!")

    cp = 1 if call_or_put == 'c' else -1
    paths = GeneratePaths(mu, sigma, S0, M, n, T)

    # Make a copy for plotting
    paths_copy = np.copy(paths)
    barrier_crossed = None

    if knockout and knockout > S0: # Up and Out
        barrier_crossed = np.any(paths > knockout, axis=0)      # checked for put (call is off)
        paths[:, barrier_crossed] = 0
    elif knockout and knockout < S0: # Down and Out
        barrier_crossed = np.any(paths < knockout, axis=0)      # checked for call (put is off)
        paths[:, barrier_crossed] = 0
    elif knockin and knockin > S0: # Up and In
        barrier_crossed = np.any(paths > knockin, axis=0)       # checked for put (0.01 diff) (call is off)
        paths[:, ~barrier_crossed] = 0
    elif knockin and knockin < S0: # Down and In
        barrier_crossed = np.any(paths < knockin, axis=0)       # checked for call (put is off)
        paths[:, ~barrier_crossed] = 0

    payoff = np.maximum(0, cp * (paths[-1] - K) [paths[-1] != 0]) # use [paths[-1] != 0] for put (cp=-1)
    option_price = np.exp(-mu * T) *  np.sum(payoff) / M

    print("Computation time is: ", round(time.time() - start_time, 4))
    #PlotPaths(paths_copy, S0, M, n, T, barrier_crossed, knockin, knockout)
    return option_price

########################################################################################
# Example usage
########################################################################################
mu = 0.05
sigma = 0.15
S0 = 100
n = 360
T = 1
K = 110
M = 50000

print("The price from Black-Scholes model is:", PriceBS(S0, K, mu, T, sigma, call_or_put= 'c'))
print("The price from Monte-Carlo method is:", PriceOption(mu, sigma, S0, M, n, T, K, call_or_put='c'))

# For a barrier option
list= []
for i in range(0,20):
    value = PriceOption(mu, sigma, S0, M, n, T, K, call_or_put='c', knockin= 125)
    list.append(value)
    print(value)
print("The price for an up and in call option is:", np.mean(list))
#print(list)

''''For box plot:'''

option_prices = []      # list of lists
mean = []               # list containing the mean value for each M
M_values = [100, 1000, 10000, 100000]
means = []
num_simulations = 20

for sim in M_values:
    prices = []
    for i in range(num_simulations):
        option_price = PriceOption(mu, sigma, S0, sim, n, T, K, call_or_put='c')
        prices.append(option_price)

    means.append(np.mean(prices))
    option_prices.append(prices)

# Plotting
plt.figure(figsize=(12, 8))

sns.boxplot(data=option_prices, width=0.5, showfliers=False, boxprops=dict(alpha=0.5))

plt.xlabel('Number of Simulations (M)', fontsize=18)
plt.ylabel('Option Price', fontsize=18)
plt.title('Monte Carlo Simulation Results for Option Pricing', fontsize=18)
plt.xticks(np.arange(len(M_values)), M_values)  # Set x-axis ticks to M_values
print(means)

plt.grid(True)
plt.show()
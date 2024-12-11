import numpy as np


def make_sinuisoid_data(
    n_time_steps=200,
    n_tokens=4,
    n_periods=3,
    composite_run=False,
    n_composite_cycles=4,
    noise=False,
):
    if n_tokens > 4:
        raise Exception
    period = n_time_steps / (2 * n_periods)
    angular_frequency = 2.0 * np.pi / period
    base_prices = (np.arange(n_tokens) + 1.0) * 5.0
    prices = np.zeros((n_time_steps, n_tokens))
    prices[:, 0] = 2 * np.cos(angular_frequency * np.arange(n_time_steps))
    prices[:, 1] = 2.5 * np.sin(angular_frequency * np.arange(n_time_steps))
    prices[:, 2] = -3.0 * np.cos(angular_frequency * np.arange(n_time_steps))
    prices += base_prices
    prices = prices[:, :n_tokens]
    if composite_run == False:
        if noise:
            prices += np.random.randn(*prices.shape)
            prices[prices < 0] = 0.1
        return prices
    elif composite_run:
        # make much slower intermediate cycles and intersperse them with the faster cycles
        slow_angular_frequency = angular_frequency / 4.0
        slow_prices = np.zeros((n_time_steps, n_tokens))
        slow_prices[:, 0] = 2 * np.cos(slow_angular_frequency * np.arange(n_time_steps))
        slow_prices[:, 1] = 2.5 * np.sin(
            slow_angular_frequency * np.arange(n_time_steps)
        )
        # slow_prices[:,2] = -3.0*np.cos(slow_angular_frequency * np.arange(n_time_steps))
        # slow_prices[:,3] = -4.0*np.sin(slow_angular_frequency * np.arange(n_time_steps))
        slow_prices += base_prices
        slow_prices = slow_prices[:, :n_tokens]
        prices = np.vstack([np.vstack([prices, slow_prices])] * n_composite_cycles)
        if noise:
            prices += np.random.randn(*prices.shape)
            prices[prices < 0] = 0.1
        return prices

<!-- PROJECT SHIELDS -->
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://quantammsim.readthedocs.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![JAX](https://img.shields.io/badge/JAX-powered-FDB515.svg)](https://github.com/google/jax)
[![Documentation Status](https://readthedocs.org/projects/quantammsim/badge/?version=latest)](https://quantammsim.readthedocs.io/en/latest/?badge=latest)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <pre>
                              _                                     _            
                             | |                                   (_)           
   __ _  _   _   __ _  _ __  | |_  __ _  _ __ ___   _ __ ___   ___  _  _ __ ___  
  / _` || | | | / _` || '_ \ | __|/ _` || '_ ` _ \ | '_ ` _ \ / __|| || '_ ` _ \ 
 | (_| || |_| || (_| || | | || |_| (_| || | | | | || | | | | |\__ \| || | | | | |
  \__, | \__,_| \__,_||_| |_| \__|\__,_||_| |_| |_||_| |_| |_||___/|_||_| |_| |_|
     | |                                                                         
     |_|                                                                         
                   
  </pre>

  <h3 align="center">quantammsim</h3>

  <p align="center">
    A Python library for simulating and tuning Automated Market Maker (AMM) protocols
    <br />
    <a href="https://quantammsim.readthedocs.io"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://quantammsim.readthedocs.io/en/latest/tutorials/getting_started.html">Getting Started</a>
    ·
    <a href="https://github.com/QuantAMMProtocol/QuantAMMSim/issues">Report Bug</a>
    ·
    <a href="https://github.com/QuantAMMProtocol/QuantAMMSim/issues">Request Feature</a>
  </p>
</div>

## About

`quantammsim` is a Python library for modeling synthetic markets, enabling modelling of Balancer, CowAMM and QuantAMM protocols. It provides tools for:

* Automated Market Making (AMM) simulation
* Arbitrage opportunity detection
* Historical data backtesting

`quantammsim` uses JAX for accelerated computation. The library focuses on dynamic AMMs that can adapt their behavior based on market conditions, with particular emphasis on Temporal Function Market Making (TFMM) pools. For more details on the theoretical foundations, see [our research](https://quantamm.fi/research).

## Features

* Multiple AMM implementations:
  * Balancer Protocol
  * CowAMM Protocol
  * QuantAMM Protocol (TFMM)
* Pre-canned textbook strategies:
  * Momentum
  * Anti-Momentum
  * Power Channel
  * Mean Reversion Channel
  * And implement custom strategies
* Include the effects of fees, gas costs, and of a provided sequence of transactions.
* JAX-accelerated computations
* Comprehensive visualization tools (see XXXXX for frontend UI)

## Installation

### Requirements
* Python 3.8+
* JAX
* NumPy
* Pandas

### Basic Installation
```bash
pip install quantammsim
```

### Development Installation
```bash
git clone https://github.com/QuantAMMProtocol/quantammsim.git
cd quantammsim
pip install -e .
```

## Quick Start

```python
from quantammsim.runners.jax_runners import do_run_on_historic_data

# Define experiment parameters
run_fingerprint = {
    'tokens': ['BTC', 'DAI'],
    'rule': 'momentum',
    'initial_pool_value': 1000000.0
}

# Run simulation
result = do_run_on_historic_data(run_fingerprint)
```

## Documentation

Full documentation is available at [quantammsim.readthedocs.io](https://quantammsim.readthedocs.io), including:

* Tutorials
* API Reference
* User Guide
* Examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/QuantAMMProtocol/quantammsim.svg?style=for-the-badge
[contributors-url]: https://github.com/QuantAMMProtocol/quantammsim/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/QuantAMMProtocol/quantammsim.svg?style=for-the-badge
[forks-url]: https://github.com/QuantAMMProtocol/quantammsim/network/members
[stars-shield]: https://img.shields.io/github/stars/QuantAMMProtocol/quantammsim.svg?style=for-the-badge
[stars-url]: https://github.com/QuantAMMProtocol/quantammsim/stargazers
[issues-shield]: https://img.shields.io/github/issues/QuantAMMProtocol/quantammsim.svg?style=for-the-badge
[issues-url]: https://github.com/QuantAMMProtocol/quantammsim/issues
[license-shield]: https://img.shields.io/github/license/QuantAMMProtocol/quantammsim.svg?style=for-the-badge
[license-url]: https://github.com/QuantAMMProtocol/quantammsim/blob/master/LICENSE

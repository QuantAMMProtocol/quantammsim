Installation
============

Requirements
------------

* Python 3.9+
* JAX
* NumPy
* Pandas

Installation
------------

It is strongly recommended to install quantammsim in a virtual environment to avoid conflicts with other packages. You can use either ``venv`` (standard Python) or ``conda``.

Using venv
^^^^^^^^^^

For Windows:

.. code-block:: bash

   python -m venv venv
   .\venv\Scripts\activate

For macOS/Linux:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate

Using Conda
^^^^^^^^^^^

Alternatively, you can use Conda to create an environment. We recommend using Python 3.10:

.. code-block:: bash

   conda create -n qsim python=3.10
   conda activate qsim

Installing the Package
^^^^^^^^^^^^^^^^^^^^^^

Once your virtual environment is activated, install the package:

.. code-block:: bash

   git clone https://github.com/QuantAMMProtocol/quantammsim.git
   cd quantammsim
   pip install -e .

To deactivate the virtual environment when you're done:

.. code-block:: bash

   deactivate

Or if you are using Conda:

.. code-block:: bash

   conda deactivate

Data Files
----------

The package can download data files for simulation and testing.
To download these files, navigate to the scripts directory first:

.. code-block:: bash

   cd scripts
   python download_data.py <tickers>


For example:

.. code-block:: bash

   cd scripts
   python download_data.py BTC ETH USDC


.. note::
   The script will automatically:
   
   * Download required files
   * Verify file integrity
   * Extract contents to the appropriate location
   * Show download and extraction progress

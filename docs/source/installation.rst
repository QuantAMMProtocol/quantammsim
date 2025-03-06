Installation
============

Requirements
------------

* Python 3.8+
* JAX
* NumPy
* Pandas

Installation
------------

It is strongly recommended to install quantammsim in a virtual environment to avoid conflicts with other packages.

For Windows:

.. code-block:: bash

   python -m venv venv
   .\venv\Scripts\activate

For macOS/Linux:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate

Once your virtual environment is activated, install the package:

.. code-block:: bash

   git clone https://github.com/QuantAMMProtocol/quantammsim.git
   cd quantammsim
   pip install -e .

To deactivate the virtual environment when you're done:

.. code-block:: bash

   deactivate

Data Files
----------

The package includes optional but recommended data files for simulation and testing.
To download these files, navigate to the scripts directory first:

.. code-block:: bash

   cd scripts
   python download_data.py

.. note::
   The data download is approximately 1.6GB. The script will automatically:
   
   * Download required files
   * Verify file integrity
   * Extract contents to the appropriate location
   * Show download and extraction progress

If the files are already present and valid, the script will skip the download.

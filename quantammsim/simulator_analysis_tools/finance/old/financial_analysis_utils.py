import numpy as np
from scipy.stats import kurtosis, skew, linregress
import pandas as pd
import csv


def convert_to_series(data, value_column):
    """
    Converts a list of dictionaries into a pandas Series indexed by date with values of avg_Wdrawdown.

    Parameters:
    - data: list of dicts, the input data where each dict contains 'date' and 'avg_Wdrawdown'.

    Returns:
    - pd.Series, the resulting Series indexed by date with values of avg_Wdrawdown.
    """

    extracted_data = [{'date': entry['date'], value_column: entry[value_column]} for entry in data]
    
    
    # Create DataFrame from the extracted data
    df = pd.DataFrame(extracted_data)
    
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set 'date' column as index
    df.set_index('date', inplace=True)
    
    series = df[value_column]
    
    return series


def arrays_to_csv(array1, array2, output_file):
    """
    Writes two arrays to a CSV file where each column is an array and each row is the corresponding elements of the arrays.

    :param array1: First array (list of values)
    :param array2: Second array (list of values)
    :param output_file: The output CSV file path (string)
    """
    if len(array1) != len(array2):
        raise ValueError("Both arrays must have the same length.")
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for a, b in zip(array1, array2):
            writer.writerow([a, b])


def dicts_to_csv(data, filename):
    # Extract keys from the first dictionary
    keys = data[0].keys()

    # Write data to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)

        # Write header
        writer.writeheader()

        # Write data rows
        for d in data:
            writer.writerow(d)


def convert_annual_to_daily_returns(rates):
  
    trading_days_per_year = 365

    if np.any(rates > 1):
        rates = rates / 100.0
        
    daily_rf_rates = (1 + rates)**(1 / trading_days_per_year) - 1

    return daily_rf_rates

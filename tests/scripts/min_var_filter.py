import json
from pathlib import Path

def filter_results(input_file, output_file=None):
    """Filter analysis results for BTC/ETH/PAXG min_variance entries.
    
    Parameters
    ----------
    input_file : str or Path
        Path to input JSON file
    output_file : str or Path, optional
        Path to save filtered results. If None, will append '_filtered' to input name
    """
    # Load results
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    # Filter for desired entries
    filtered_results = [
        result for result in results
        if (result["tokens"] == ["BTC", "ETH", "PAXG"] and 
            result["rule"] == "min_variance")
    ]
    
    # Determine output path
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
    
    # Save filtered results
    with open(output_file, 'w') as f:
        json.dump(filtered_results, f, indent=2)
    
    print(f"Found {len(filtered_results)} matching entries")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    filter_results("experiments/best_trial_analysis/all_analysis_results_sharpe_Sun26thJan2025_.json")
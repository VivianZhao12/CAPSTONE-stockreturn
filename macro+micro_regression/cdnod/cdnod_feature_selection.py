import pandas as pd
import os
import json

# Define directories
CDNOD_DIR = "cdnod_graph"
TICKERS = ["t", "goog", "amzn", "amgn", "abt", "cvs"]

causal_features = {}

for ticker in TICKERS:
    edges_path = os.path.join(CDNOD_DIR, f'{ticker.lower()}_fisherz_M.csv')
    
    if not os.path.exists(edges_path):
        print(f"Warning: {edges_path} not found, skipping {ticker}.")
        continue
    
    edges = pd.read_csv(edges_path)
    causal_graph = {}
    
    for _, row in edges.iterrows():
        cause, effect = row["cause"], row["effect"]
        if effect not in causal_graph:
            causal_graph[effect] = []
        causal_graph[effect].append(cause)
    
    # Step 1: Find direct causes of "Close"
    direct_causes = set(causal_graph.get("Close", []))
    
    # Step 2: Find indirect causes recursively
    indirect_causes = set()
    
    def find_indirect_causes(current_vars):
        """ Recursively finds indirect causes by checking parents of current_vars """
        new_vars = set()
        for var in current_vars:
            if var in causal_graph:  # Check if it has parents
                parents = causal_graph[var]
                for parent in parents:
                    if parent not in direct_causes and parent not in indirect_causes:  # Avoid duplicates
                        new_vars.add(parent)
        
        if new_vars:  # If new indirect causes are found, continue searching deeper
            indirect_causes.update(new_vars)
            find_indirect_causes(new_vars)
    
    # Start recursive search for indirect causes
    find_indirect_causes(direct_causes)
    
    # Store results in dictionary
    causal_features[ticker.upper()] = list(direct_causes) + list(indirect_causes)

# Save as JSON
output_path = os.path.join(CDNOD_DIR, "causal_feature.json")
with open(output_path, "w") as json_file:
    json.dump(causal_features, json_file, indent=4)
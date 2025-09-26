import wandb
import pandas as pd

# Initialize the wandb API
api = wandb.Api()

# Specify the run path
run_path = "cyhsm/distill-nemotron-sep25/mfw5i67c"
test_against = "mean_token_accuracy"
#test_against = "sft_loss"
# test_against = "entropy"

try:
    # Get the run object
    run = api.run(run_path)

    # Get the run history as a pandas DataFrame
    # We are interested in the 'train/sft_loss' and 'train/layer_N_loss' metrics
    keys = [f"train/{test_against}"] + [f"train/layer_{i}_loss" for i in range(32)]
    history = run.history(keys=keys)

    # Calculate the correlations
    correlations = {}
    sft_loss = history[f"train/{test_against}"]

    for i in range(32):
        layer_loss_key = f"train/layer_{i}_loss"
        if layer_loss_key in history.columns:
            layer_loss = history[layer_loss_key]
            # Calculate the Pearson correlation coefficient
            correlation = sft_loss.corr(layer_loss)
            correlations[i] = correlation

    # Find the layer with the highest absolute correlation
    if correlations:
        best_layer = max(correlations, key=lambda layer: abs(correlations[layer]))
        best_correlation = correlations[best_layer]

        print(f"Correlations between layer losses and sft_loss:")
        for layer, corr in correlations.items():
            print(f"  Layer {layer}: {corr:.4f}")

        print(f"\nRun path: {run_path}")
        print(f"\nThe layer with the highest absolute correlation with {test_against} is Layer {best_layer} with a correlation of {best_correlation:.4f}")
    else:
        print("No layer loss data found to correlate.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure the run path is correct and you have the necessary permissions to access the run.")
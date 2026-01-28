import wandb

api = wandb.Api()
run = api.run("cyhsm/loom/zhvqr1sl")

# Get all metric names from the summary
metric_names = list(run.summary.keys())

print(metric_names)
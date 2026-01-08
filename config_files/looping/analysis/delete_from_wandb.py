import wandb

api = wandb.Api()
runs = api.runs("cyhsm/loop")

for run in runs:
    for file in run.files():
        if "output.log" in file.name:
            print(f"Cleaning logs for run: {run.name}")
            file.delete()
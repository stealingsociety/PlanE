import wandb

# Log in if needed
wandb.login()

# Replace with your username
username = "aliajrigers-desydeutsches-elektronen-synchrotron"
project = "PlanE"

api = wandb.Api()

# Get all sweeps in the project
sweeps = api.sweeps(f"{username}/{project}")

for sweep in sweeps:
    if sweep.state == "running":
        print(f"Stopping sweep: {sweep.name} ({sweep.id})")
        sweep.update(state="finished")

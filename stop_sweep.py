import wandb

# Log in if needed
wandb.login()

# Replace with your username
username = "aliajrigers-desydeutsches-elektronen-synchrotron"
project = "PlanE"

api = wandb.Api()

# Initialize API
api = wandb.Api()

# Get all sweeps for this project
sweeps = api.sweeps(f"{entity}/{project}")

# Cancel all running sweeps
for sweep in sweeps:
    if sweep.state == "running":
        print(f"Cancelling sweep {sweep.id}")
        sweep.cancel()



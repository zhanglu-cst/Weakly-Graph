import wandb


def func():
    for i in range(20):
        wandb.log({"loss": i + 2})

# Optional
# wandb.watch(model)

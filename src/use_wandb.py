import wandb

def use_wandb(params: dict, PROJECT_NAME:str, TRIAL_NAME: str):
    # wandb.login()

    wandb.init(
        config=params,
        project=PROJECT_NAME,
        name=TRIAL_NAME
    )
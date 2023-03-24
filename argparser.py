import os
from typing import Optional

from tap import Tap


class PPOGridNetCoacaiNoMask(Tap):
    # Common arguments
    exp_name: Optional[str] = os.path.basename(__file__).rstrip(".py")
    gym_id: Optional[str] = "MicrortsDefeatCoacAIShaped-v3"
    learning_rate: Optional[float] = 2.5e-4
    seed: Optional[int] = 1
    total_timesteps: Optional[int] = 1e8
    torch_deterministic: Optional[bool] = True
    cuda: Optional[bool] = True
    prod_mode: Optional[bool] = True
    capture_video: Optional[
        bool
    ] = False  # whether to capture videos of the agent performances (check out `videos` folder)
    wandb_project_name: Optional[str] = "cleanRL"
    wandb_entity: Optional[str] = None

    # Algorithm specific arguments

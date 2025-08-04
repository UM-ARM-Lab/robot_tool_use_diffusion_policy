"""
Debug training script that runs without command line arguments.
Usage: python train_debug.py
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=".",
    config_name="victor_diffusion_policy_state_clean_test"
)
def main(cfg: OmegaConf):
    # print('the cfg is:')
    print(OmegaConf.to_yaml(cfg))
    # Apply debug settings
    cfg.training.debug = False
    cfg.training.seed = 7
    cfg.training.device = "cuda:0"
    
    # Update dataset path to use the actual file (absolute path)
    folder_path = "/home/yatin/Documents/Projects/forceful_tool_use/diffusion_related/robot_tool_use_diffusion_policy/data/victor/"
    # folder_path = "/home/yatin/Documents/Wolverine/Research/force_tool_acoustic/diffusion_related/robot_tool_use_diffusion_policy/data/victor/"

    # cfg.task.dataset.zarr_path = folder_path +  "dataset_2025-07-21_13-07-55.zarr.zip"
    cfg.task.dataset.zarr_path = folder_path +  "dspro_07_22_no_wrench.zarr.zip"

    # cfg.task.dataset.zarr_path = "/data/victor/traj_1.zarr"
     # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    
    print("=== DEBUG MODE ENABLED ===")
    print(f"Debug mode: {cfg.training.debug}")
    print(f"Device: {cfg.training.device}")
    print(f"Seed: {cfg.training.seed}")
    print(f"Dataset path: {cfg.task.dataset.zarr_path}")
    print("Full dataset config:")
    print(OmegaConf.to_yaml(cfg.task.dataset))
    print("===========================")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

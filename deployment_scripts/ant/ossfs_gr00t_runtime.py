from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_OSSFS_WORKSPACE = Path("/home/jetson/Desktop/project/ossfs/node_59823209/workspace")


@dataclass(frozen=True)
class OssfsGr00tRuntime:
    workspace: Path
    gr00t: Any
    LeRobotSingleDataset: Any
    DATA_CONFIG_MAP: dict[str, Any]
    load_data_config: Any | None
    Gr00tPolicy: Any
    COMPUTE_DTYPE: Any
    unsqueeze_dict_values: Any


def _ensure_ossfs_gr00t_on_path() -> Path:
    workspace = str(_OSSFS_WORKSPACE)
    sys.path[:] = [path for path in sys.path if path != workspace]
    sys.path.insert(0, workspace)
    return _OSSFS_WORKSPACE


def _import_local_gr00t_runtime() -> OssfsGr00tRuntime:
    _ensure_ossfs_gr00t_on_path()
    for module_name in list(sys.modules):
        if module_name == "gr00t" or module_name.startswith("gr00t."):
            del sys.modules[module_name]

    gr00t = importlib.import_module("gr00t")
    if not Path(gr00t.__file__).resolve().is_relative_to(_OSSFS_WORKSPACE.resolve()):
        raise RuntimeError(f"Expected local ossfs gr00t, got {gr00t.__file__}")

    dataset_mod = importlib.import_module("gr00t.data.dataset")
    data_config_mod = importlib.import_module("gr00t.experiment.data_config")
    policy_mod = importlib.import_module("gr00t.model.policy")
    return OssfsGr00tRuntime(
        workspace=_OSSFS_WORKSPACE,
        gr00t=gr00t,
        LeRobotSingleDataset=dataset_mod.LeRobotSingleDataset,
        DATA_CONFIG_MAP=data_config_mod.DATA_CONFIG_MAP,
        load_data_config=getattr(data_config_mod, "load_data_config", None),
        Gr00tPolicy=policy_mod.Gr00tPolicy,
        COMPUTE_DTYPE=getattr(policy_mod, "COMPUTE_DTYPE", None),
        unsqueeze_dict_values=policy_mod.unsqueeze_dict_values,
    )

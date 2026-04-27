import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from deployment_scripts.ant import ossfs_gr00t_runtime as module


class OssfsGr00tRuntimeImportTest(unittest.TestCase):
    def test_import_local_gr00t_runtime_accepts_missing_load_data_config(self):
        fake_workspace = Path("/tmp/fake-ossfs-workspace")

        fake_gr00t = types.ModuleType("gr00t")
        fake_gr00t.__file__ = str(fake_workspace / "gr00t" / "__init__.py")

        fake_dataset_mod = types.ModuleType("gr00t.data.dataset")
        fake_dataset_mod.LeRobotSingleDataset = object()

        fake_data_config_mod = types.ModuleType("gr00t.experiment.data_config")
        fake_data_config_mod.DATA_CONFIG_MAP = {"new_interaction_group": object()}

        fake_policy_mod = types.ModuleType("gr00t.model.policy")
        fake_policy_mod.Gr00tPolicy = object()
        fake_policy_mod.unsqueeze_dict_values = lambda value: value

        fake_modules = {
            "gr00t": fake_gr00t,
            "gr00t.data.dataset": fake_dataset_mod,
            "gr00t.experiment.data_config": fake_data_config_mod,
            "gr00t.model.policy": fake_policy_mod,
        }

        with (
            mock.patch.object(module, "_OSSFS_WORKSPACE", fake_workspace),
            mock.patch.object(module, "_ensure_ossfs_gr00t_on_path", return_value=fake_workspace),
            mock.patch.object(
                module,
                "sys",
                SimpleNamespace(path=[], modules={"gr00t": object(), "gr00t.model": object()}),
            ),
            mock.patch.object(
                module.importlib,
                "import_module",
                side_effect=lambda name: fake_modules[name],
            ),
        ):
            runtime = module._import_local_gr00t_runtime()

        self.assertEqual(runtime.workspace, fake_workspace)
        self.assertEqual(runtime.DATA_CONFIG_MAP, {"new_interaction_group": mock.ANY})
        self.assertIsNone(runtime.load_data_config)
        self.assertIs(runtime.LeRobotSingleDataset, fake_dataset_mod.LeRobotSingleDataset)
        self.assertIs(runtime.Gr00tPolicy, fake_policy_mod.Gr00tPolicy)


if __name__ == "__main__":
    unittest.main()

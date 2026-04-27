# New Interaction Group Config Design

**Goal:** Make the current branch understand the single-arm `new_embodiment` export shape by adding the missing built-in data config entries already validated in `ossfs` and `.worktrees`.

**Recommended approach:** Merge the minimal `NewInteractionGroup` and `NewInteractionGroupFuture` config classes into the current [`gr00t/experiment/data_config.py`](/home/jetson/Desktop/project/Isaac-GR00T/gr00t/experiment/data_config.py) and register them in `DATA_CONFIG_MAP`. This keeps `export_onnx.py` usage simple, avoids passing a separate external module path, and reuses an implementation already present in adjacent project copies.

**Rejected alternatives:**
- Copy a standalone `custom_data_config.py`: lower blast radius, but it complicates the export command and leaves the current branch inconsistent with the validated local variants.
- Replace the whole file from `ossfs` or a worktree: faster, but it risks importing unrelated behavior changes into the current branch.

**Scope:** Only add the two missing data config classes and their map entries. Do not pull unrelated worktree changes into the branch.


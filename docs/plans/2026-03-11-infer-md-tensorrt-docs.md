# Infer.md TensorRT Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update `infer.md` so the Jetson Orin TensorRT path includes the missing conda/TensorRT Python binding setup, validation commands, and troubleshooting flow.

**Architecture:** Keep the existing PyTorch-first document structure, but expand the TensorRT section into a linear workflow: prerequisite checks, conda binding setup, ONNX export, engine build, and inference verification. Add a matching troubleshooting entry so the common `ModuleNotFoundError: No module named 'tensorrt'` failure is explained at the point of failure and again in the FAQ.

**Tech Stack:** Markdown, Jetson Orin, Conda, TensorRT 10.7, Python 3.10

---

### Task 1: Add the missing TensorRT prerequisite flow

**Files:**
- Modify: `/home/jetson/Desktop/project/dox/infer.md`

**Step 1: Expand the TensorRT overview**

Explain that PyTorch success does not guarantee TensorRT Python bindings are visible inside `gr00t-orin`, and that the failure happens during `import tensorrt` before engine loading.

**Step 2: Add prerequisite verification commands**

Document commands for:
- checking installed Jetson TensorRT packages with `dpkg -l`
- validating TensorRT import under `/usr/bin/python3`
- validating TensorRT import inside the `gr00t-orin` conda environment

**Step 3: Add temporary and permanent conda fixes**

Document:
- temporary `PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH`
- permanent `.pth` file under `$CONDA_PREFIX/lib/python3.10/site-packages/`

**Step 4: Reorder the remaining TensorRT flow**

After the conda binding section, keep:
- ONNX export
- engine build
- TensorRT inference

Add a short verification command before the build/inference steps.

### Task 2: Update the execution order and troubleshooting sections

**Files:**
- Modify: `/home/jetson/Desktop/project/dox/infer.md`

**Step 1: Update the recommended execution order**

Insert the TensorRT prerequisite/binding steps between PyTorch validation and ONNX export.

**Step 2: Add a troubleshooting entry for `No module named 'tensorrt'`**

State root cause clearly:
- Jetson system package exists
- conda environment does not include `/usr/lib/python3.10/dist-packages`

Then provide the same temporary/permanent fixes and a verification command.

**Step 3: Renumber the later troubleshooting entries**

Keep the rest of the FAQ intact while shifting numbering and preserving readability.

### Task 3: Verify document consistency

**Files:**
- Modify: `/home/jetson/Desktop/project/dox/infer.md`

**Step 1: Re-read edited sections**

Check that section numbers, command paths, and TensorRT package paths are internally consistent.

**Step 2: Confirm commands match local evidence**

Ensure the documentation reflects the verified local state:
- TensorRT Python package at `/usr/lib/python3.10/dist-packages/tensorrt`
- system import works under `/usr/bin/python3`
- conda import works after adding the path

**Step 3: Keep scope tight**

Do not add unrelated TensorRT optimization guidance or rebuild code changes.

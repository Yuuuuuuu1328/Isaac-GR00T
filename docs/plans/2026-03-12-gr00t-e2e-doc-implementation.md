# GR00T E2E Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write a detailed GR00T end-to-end inference document for the repository's current default path and place a copy at `/home/jetson/Desktop/project/dox/gr00t_e2e.md`.

**Architecture:** Author the markdown inside the repository so it is versionable and editable with normal repo tooling, then copy the finished file to the user-requested external path. Base every section on verified repository source and verified sample shapes rather than assumptions.

**Tech Stack:** Markdown, repository source inspection, shell verification, `apply_patch`

---

### Task 1: Freeze the documentation scope

**Files:**
- Create: `docs/plans/2026-03-12-gr00t-e2e-doc-design.md`

**Step 1: Write the scope and target path**

Describe the exact inference path to document:

- `GR00T-N1.5-3B`
- `fourier_gr1_arms_only`
- single ego-view image
- action horizon `16`
- denoising steps `4`

**Step 2: Record required content**

List the required sections:

- raw input
- transforms
- tensor shapes
- backbone
- action head
- repeated requests
- output reconstruction
- acceleration analysis

### Task 2: Draft the repository markdown

**Files:**
- Create: `docs/gr00t_e2e.md`

**Step 1: Write the high-level explanation**

Explain what a task is and how repeated inference requests happen under one task instruction.

**Step 2: Write the verified input/output section**

Use the verified demo sample shapes and explain which fields are raw observation, which are model-ready tensors, and which are final unnormalized actions.

**Step 3: Write the step-by-step tensor flow**

Cover:

- `Gr00tPolicy.get_action`
- modality transform chain
- `GR00T_N1_5.prepare_input`
- `EagleBackbone.forward`
- `FlowmatchingActionHead.get_action`
- `unapply_transforms`

**Step 4: Write the acceleration analysis**

Separate:

- low-risk practical optimizations
- medium-risk optimizations
- changes that are not recommended

### Task 3: Copy the markdown to the user path

**Files:**
- Copy from: `docs/gr00t_e2e.md`
- Copy to: `/home/jetson/Desktop/project/dox/gr00t_e2e.md`

**Step 1: Create the destination directory if missing**

Run:

```bash
mkdir -p /home/jetson/Desktop/project/dox
```

**Step 2: Copy the file**

Run:

```bash
cp docs/gr00t_e2e.md /home/jetson/Desktop/project/dox/gr00t_e2e.md
```

### Task 4: Verify the generated document

**Files:**
- Verify: `docs/gr00t_e2e.md`
- Verify: `/home/jetson/Desktop/project/dox/gr00t_e2e.md`

**Step 1: Check both files exist**

Run:

```bash
ls -l docs/gr00t_e2e.md /home/jetson/Desktop/project/dox/gr00t_e2e.md
```

Expected:

- both files exist
- non-zero size

**Step 2: Inspect the key sections**

Run:

```bash
rg -n "^#|^##" docs/gr00t_e2e.md
```

Expected:

- headers for overview, inputs, transforms, model path, repeated requests, acceleration analysis

**Step 3: Inspect the copied file head**

Run:

```bash
sed -n '1,80p' /home/jetson/Desktop/project/dox/gr00t_e2e.md
```

Expected:

- content matches the repository draft
- markdown starts with the intended title and overview

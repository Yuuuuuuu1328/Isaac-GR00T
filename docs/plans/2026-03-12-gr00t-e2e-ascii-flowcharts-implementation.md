# GR00T E2E ASCII Flowcharts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add plain-text flowcharts to the GR00T end-to-end documentation and sync the updated file to `/home/jetson/Desktop/project/dox/gr00t_e2e.md`.

**Architecture:** Keep the existing markdown as the source of truth and insert a few fixed-width ASCII diagrams at the highest-value points in the document. After updating the repository copy, overwrite the requested external copy and verify both the file structure and destination content.

**Tech Stack:** Markdown, ASCII diagrams, shell verification, `apply_patch`

---

### Task 1: Document the flowchart design

**Files:**
- Create: `docs/plans/2026-03-12-gr00t-e2e-ascii-flowcharts-design.md`

**Step 1: Record the scope**

Write the design note covering:

- top-level end-to-end flowchart
- action-head denoising loop flowchart
- repeated-request timeline flowchart

**Step 2: Record constraints**

Explicitly state that Mermaid is not available and the diagrams must render as plain text.

### Task 2: Update the repository markdown

**Files:**
- Modify: `docs/gr00t_e2e.md`

**Step 1: Add a top-level request flowchart**

Insert a fixed-width ASCII overview near the start of the document.

**Step 2: Add an action-head denoising loop flowchart**

Place it in the action-head section close to the per-step explanation.

**Step 3: Add a repeated-request timeline flowchart**

Place it in the multi-request section to show one task with multiple `policy.get_action()` calls.

### Task 3: Sync the external copy

**Files:**
- Copy from: `docs/gr00t_e2e.md`
- Copy to: `/home/jetson/Desktop/project/dox/gr00t_e2e.md`

**Step 1: Overwrite the destination**

Run:

```bash
cp docs/gr00t_e2e.md /home/jetson/Desktop/project/dox/gr00t_e2e.md
```

### Task 4: Verify the result

**Files:**
- Verify: `docs/gr00t_e2e.md`
- Verify: `/home/jetson/Desktop/project/dox/gr00t_e2e.md`

**Step 1: Check the new flowchart sections**

Run:

```bash
rg -n "流程图|时间线|去噪循环" docs/gr00t_e2e.md
```

Expected:

- the new ASCII flowchart section titles are present

**Step 2: Check both file sizes**

Run:

```bash
ls -l docs/gr00t_e2e.md /home/jetson/Desktop/project/dox/gr00t_e2e.md
```

Expected:

- both files exist
- destination size matches repository size

**Step 3: Inspect the updated destination copy**

Run:

```bash
sed -n '1,140p' /home/jetson/Desktop/project/dox/gr00t_e2e.md
```

Expected:

- the top-level ASCII flowchart appears near the beginning of the copied file

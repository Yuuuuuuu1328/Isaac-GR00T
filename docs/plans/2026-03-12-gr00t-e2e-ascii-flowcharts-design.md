# GR00T E2E ASCII Flowcharts Design

**Goal**

Supplement the existing `docs/gr00t_e2e.md` document with plain-text flowcharts that remain readable in Markdown environments without Mermaid support.

**Approach**

Keep the current narrative intact and add a small number of high-signal diagrams:

1. A top-level single-request end-to-end flowchart near the beginning of the document.
2. A per-step action-head denoising loop flowchart in the action-head section.
3. A repeated-request timeline flowchart in the multi-request section.

The diagrams should use fixed-width ASCII blocks and arrows so they render consistently in terminals, code hosts, and editors.

**Constraints**

- Do not rely on Mermaid or external rendering.
- Do not restructure the whole document.
- Keep the diagrams aligned with the verified default path:
  - `GR00T-N1.5-3B`
  - `fourier_gr1_arms_only`
  - `state_horizon=1`
  - `action_horizon=16`
  - `num_inference_timesteps=4`

**Non-Goals**

- No image assets.
- No SVG or HTML.
- No attempt to diagram every internal helper function.

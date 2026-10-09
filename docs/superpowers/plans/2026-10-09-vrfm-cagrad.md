# OpenWAM VRFM + CAGrad implementation

User-approved design: the 2026-10-09 framework figure and follow-up mechanism discussion. The user explicitly requested implementation in the existing WAM repository. Implement on `feat/vrfm-cagrad` from `origin/main` f85b35c; do not alter external training, frozen round02 assets, or resume supervision.

## Contract

- Retain the native OpenWAM pipeline, video/action experts, masks, noise schedules, and data contracts. Add no private K/V, routing gates, or extra policy experts to the new method.
- VRFM: training-only Gaussian posterior over a continuous shared z, conditioned on the paired clean/noisy video/action inputs, time, and available context. Reparameterization, analytic KL to fixed N(0,I), shared conditioning of both native streams. Inference and heldout proxies sample the fixed prior without target-dependent posterior; hold one z fixed over each generation trajectory. Default latent_dim=32 and beta=0.001 are provisional configurable engineering defaults, not validated hyperparameters.
- Decoder exposes differentiable weighted video/action losses separately in `aux['task_losses']`, keys `video` and `action`, plus weighted KL in `aux['vrfm_kl_loss']`. Existing reported metrics may remain detached. Posterior parameters receive the ordinary sum of these terms.
- CAGrad only coordinates generator parameters receiving both task gradients; parameters belonging to one task retain that task's gradient, and the posterior is excluded. Use the same forward sample for both gradients. Accumulate task gradients over the whole optimizer batch and average across distributed ranks BEFORE solving CAGrad. Apply clipping once after composition, then AdamW. Default c=0.4, no hidden task normalization or third KL task. CAGrad gives no AdamW/control non-degradation guarantee.
- Four public arms: baseline, vrfm, cagrad, vrfm_cagrad. Clearly mark old v0.2 routes as legacy; new defaults and quickstarts must never silently enable them. Keep old historical contracts/checkpoints identifiable and reject incompatible resume.
- All four arms explicitly train the native action expert/packed action blocks and the last native video block's self-attention K/V, plus VRFM modules when enabled. Freeze other video parameters. Record this scope in checkpoint identity and audit; attaching VRFM must not override existing trainability. This is a bounded adaptation, not full video-backbone fine-tuning.

## Ownership and tasks

1. Algorithm engineer: implement and test the standalone two-task CAGrad solver in `src/gradientwam/cagrad.py`, including official-code provenance/license and degenerate cases. Interface `cagrad_coefficients(gram, c=0.4)` returns two scalar coefficients for the coordinated gradient, using the 2x2 task Gram matrix. The trainer applies these only to its common-parameter subset. Produce a concise mathematical audit for VRFM and CAGrad.
2. Infra engineer: own `src/open_wam/**` model/decoder integration and focused `tests/test_vrfm*.py`. Public attachment `configure_vrfm(pipeline, latent_dim=32, kl_weight=0.001)` before optimizer construction; expose posterior parameter identification helper. Provide native tiny training/inference evidence, no target leakage and no private parameters.
3. Training engineer: own `src/gradientwam/**` except `cagrad.py`, plus native training files only by explicit coordination with infra; train/smoke/resume/DDP integration, method settings, and `tests/test_gradientwam_vrfm_cagrad*.py`. Provide global-gradient and accumulation equivalence tests, strict resume identity, prior-only heldout evaluation. Preserve 1/2/8-rank launch semantics. No GPU launch or remote asset mutations.
4. PM: integrate interfaces, new YAML arms/default scripts, README/method/validation docs, attribution, focused end-to-end tests, independent review, publish a reviewable branch/PR. Existing employee chats only; no new subagents/chats. No full-size training or new asset downloads.

## Review focus

- CAGrad zero/opposed/collinear gradients, c=0 and invalid/nonfinite inputs.
- Gradients accumulated across microbatches/ranks before the nonlinear solve; DDP does not reduce the same gradient twice.
- Posterior receives reconstruction and KL gradients, generator receives no target information during prior-only evaluation/inference.
- z is fixed across ODE calls and checkpoint recomputation; prior sampling and RNG resume are reproducible.
- Public quickstart selects the new method; old routes stay explicitly legacy; configs and full-state checkpoints cannot silently cross methods.

## Verification

Each owner writes/runs focused failing checks before implementation and reports actual commands/results. Run meaningful tiny CPU forward/backward, numeric CAGrad parity, two-rank CPU distributed parity, checkpoint restoration, config/CLI checks and relevant existing tests. Full-width GPU and actual eight-GPU/control results remain unverified unless separately authorized and run. Do not present engineering tests as research gains.

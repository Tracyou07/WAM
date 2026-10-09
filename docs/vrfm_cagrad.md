# VRFM + CAGrad on OpenWAM

This is the current GradientWAM method. It replaces the earlier shared/private K/V hypothesis; no additional private attention path or policy expert is part of this design. Implementation correctness and task performance are separate questions.

## Forward model and variational objective

The native OpenWAM video and action streams keep their masks, state/action schema, and noise schedules. A Gaussian posterior q_phi(z | clean/noisy video, clean/noisy action, time, context) is used only during training. It produces a mean and diagonal variance; the reparameterized sample z = mu + sigma * epsilon conditions both streams.

The prior is fixed N(0,I), not a learned observation-conditioned route prior. The losses are the native weighted video/action flow-matching objectives plus beta * KL(q_phi || N(0,I)). KL is computed analytically and differentiably. The posterior is optimized using the ordinary sum of reconstruction and KL gradients. The generator's conditioning projections are model parameters, not posterior parameters.

Default engineering settings are latent_dim=32 and beta=0.001. They are configurable starting values, not validated research choices. This adapts the VRFM idea to paired video/action data and OpenWAM's native noise/time conventions; it does not claim exact reproduction of the original image-generation experiments or transfer of all their theoretical results.

Heldout denoising calls put the model in evaluation mode. They may use heldout targets to construct an evaluation loss, but must never encode those targets into z. Real generation always samples the fixed prior, even if the caller forgot to change train/eval mode. One z remains fixed over a generated chunk/ODE trajectory, including conditioned/unconditioned evaluations and activation-checkpoint recomputation.

## Gradient coordination

Let g_v and g_a be gradients of the **already weighted** native video and action objectives on generator parameters reached by both tasks. CAGrad chooses d by maximizing min(g_v dot d, g_a dot d), subject to ||d - g_0|| <= c ||g_0|| and g_0=(g_v+g_a)/2.

The two-task solver works from the 2x2 Gram matrix. Its coefficients form the original, unrescaled CAGrad direction; they are not probabilities and need not sum to one. The default c=0.4 is configurable in [0,1). At c=0 the solver returns the mean task gradient. Native total-loss training uses a sum; gradient magnitude is therefore part of this algorithmic change and must be reported when interpreting comparisons.

The training order is:

1. Use the same forward graph, samples, masks, noise and z for both losses.
2. Accumulate each task's gradients over the whole optimizer batch.
3. Average the task-gradient buffers across ranks.
4. Form the global common-parameter Gram matrix and solve CAGrad once.
5. Install the coordinated gradient on common generator parameters. Other generator parameters retain their ordinary task gradients. Posterior parameters retain ordinary video + action + KL gradients; KL is not a third CAGrad task.
6. Clip the assembled gradient once, then call AdamW and the scheduler.

Solving separately on each rank or on each microbatch and then averaging is a different nonlinear algorithm. The public distributed path is DDP, not FSDP. No post-update acceptance/rejection mechanism is included.

CAGrad treats the two tasks symmetrically. Its local gradient geometry does not establish action-priority protection, an AdamW displacement guarantee, or robot success-rate improvement. VRFM is hypothesized to reduce ambiguity-related interference; CAGrad coordinates remaining optimization conflict. Both claims need paired research evidence.

## Public controls and provenance

The public 2x2 is baseline, VRFM, CAGrad, VRFM+CAGrad. The VRFM pair has identical added architecture, and the CAGrad pair shares the same solver/settings. Keep the same native initialization, trainability, process count, episode split, optimizer budget and noise recipe. Baseline and VRFM do not have identical parameter counts.

The default trainable subset is the native action expert and packed action blocks, plus the last video block's native self-attention K/V projections. Other video weights are frozen. VRFM adds trainable posterior/conditioning projections without changing that subset. The wrapper applies and records this same scope in all four arms; it is a resource-conscious adaptation, not full video-backbone fine-tuning. DDP still stores all frozen model weights on every rank.

With the enforced `decoupled_same_step` / `video_only` attention contract, video queries cannot read action keys. The structural CAGrad candidates are therefore the final video K/V projections and, for VRFM, the video latent projection. The action latent projection and native action parameters keep their ordinary gradients. Task activity and graph connectivity determine participation within this candidate set; a numerically zero gradient does not remove a shared parameter. Changing the attention contract requires revisiting this structural scope.

The new settings live under the root `gradientwam` mapping, outside the native OpenWAM experiment schema. Method/latent/KL/CAGrad settings belong in the checkpoint identity. Old v0.2 arms require `legacy_v02: true` and cannot silently resume as the new method.

Reproducible resume within an arm does not imply identical stochastic tensors across arms: VRFM adds latent draws to the Torch RNG stream. Equal seeds alone do not establish a strict paired-noise experiment. Materialize or separately control evaluation noise/time draws before reporting paired research comparisons. The current posterior summarizes video/action statistics and context; it is a compact adaptation, not the original paper's encoder architecture.

See [README](../README.md) for commands, [model handoff](vrfm_model_handoff.md) and [training handoff](vrfm_training_handoff.md) for implementation details, and [validation](validation.md) for commands actually run.

## Primary references

- Guo and Schwing. [Variational Rectified Flow Matching, ICML 2025](https://proceedings.mlr.press/v267/guo25i.html).
- Liu et al. [Conflict-Averse Gradient Descent for Multi-task Learning, NeurIPS 2021](https://arxiv.org/abs/2110.14048). [Official implementation](https://github.com/Cranial-XIX/CAGrad).

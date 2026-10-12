# Matched GradientWAM DDP comparison and resume

The user-approved comparison is the project's bounded OpenWAM baseline versus
the same model with VRFM and CAGrad. It supersedes the earlier full-backbone FSDP
proposal. Both existing data entries remain supported.

- Train the native action expert, packed action blocks, final video self-attention
  K/V, and the native hidden proprioceptive-state projection in both arms.
- Only VRFM arms add a posterior and latent projections. Initialize those modules
  independently without changing the native initialization RNG stream.
- Keep the public recipes matched in native initialization, data, preprocessing,
  interaction, AdamW recipe, train seed, evaluation seed, and DDP topology.
- Keep actual entry defaults at two optimizer updates. Ten thousand updates are
  an explicit future run setting, not the current default.
- Preserve full-state continuation: model, optimizer, scheduler, strategy/scaler,
  step state, per-rank RNG, and sampling cursor. Reject changed method/scope/data/
  topology on resume. Preserve legacy bounded checkpoints with their original
  settings; do not silently reinterpret them as state-projection training.
- Exclude posterior parameters from CAGrad. Include the state projection only
  when both task graphs actually reach it.
- Preserve the historical v0.2 specification and unrelated local modifications.

Validation must show actual state-projection updates and identical native
parameter scope across arms, then compare interrupted and uninterrupted DDP
VRFM+CAGrad training, including optimizer and scheduler state. A successful
engineering check is not a claim of improved LIBERO success or full-width GPU fit.

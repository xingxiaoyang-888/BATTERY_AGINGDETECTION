# Research model card

## Intended use and evidence

Battery aging research, architecture comparisons, and engineering scenario exploration. This release provides executable implementations and structural tests, **not trained checkpoints or demonstrated accuracy gains**. Synthetic UI data is not experimental evidence. The UI never invokes an optimizer.

## Encoders and references

**Patch Transformer:** histories `[B,T,C]` become overlapping temporal patches independently per channel. Channels share embedding/attention weights; positional embeddings encode order. Pooled channel representations are concatenated and projected. This follows the channel-independent patch idea of [PatchTST](https://arxiv.org/abs/2211.14730), with a custom battery fusion/head rather than exact paper reproduction.

**Inverted Transformer:** each variable's history becomes one token; attention learns dependencies between variables. Inspired by [iTransformer](https://arxiv.org/abs/2310.06625), with the common battery degradation head replacing the generic forecasting head.

**Selective state space:** causal depthwise convolution extracts local features. Input-dependent step sizes and B/C coefficients control a diagonal state update. Decay is `exp(-softplus(delta) * exp(log_A))`, which stays between zero and one. Gating and residual blocks complete a linear-time temporal scan inspired by [Mamba](https://arxiv.org/abs/2312.00752). This Python-loop reference is not the official selective scan and does not implement the structured duality/fused kernels of [Mamba-2](https://proceedings.mlr.press/v235/dao24a.html).

**TCN:** causally padded, exponentially dilated residual convolution blocks provide an efficient deep baseline. Layer normalization acts across channels at each timestep.

## Physical quantile head

Softplus produces three positive rate components. Successive positive sums form slow, middle, and fast rates. The resulting trajectories are:

```text
fast_rate >= middle_rate >= slow_rate >= 0
q10[t] = current_soh - sum(fast_rate[:t])
q50[t] = current_soh - sum(middle_rate[:t])
q90[t] = current_soh - sum(slow_rate[:t])
```

Clipping to `[0,105]` percent preserves ordering and nonincrease. These are **monotonic physical priors**, not a complete electrochemical PINN. Temporary capacity recovery is intentionally excluded from the projected degradation trajectory. Quantile labels gain statistical meaning only after training and validation.

## Objectives

- Pinball loss supervises q10/q50/q90 against complete future SOH windows.
- Patch Transformer supports a masked reconstruction auxiliary loss. Raw timesteps are masked **before** overlapping patch formation, preventing neighboring patches from revealing exact masked values. This is joint auxiliary learning, not a pretrained foundation model.
- A discrete hazard head combines event likelihood for observed EOL crossings with survival likelihood for right-censored horizons. Samples already below EOL are excluded. Censored windows never receive invented event times.
- AdamW, cosine scheduling, deterministic seeds, gradient clipping, and validation-based early stopping are implemented.

The window builder currently requires complete SOH targets. It handles EOL censoring inside these windows, but does not optimize truncated SOH windows at the end of a history. The hazard head is an auxiliary signal; it is not served as a calibrated RUL distribution.

## Splitting and scaling

Seeded cell-level partitions isolate train, validation, calibration, and test data. At least 20 cells are required, including at least four calibration cells. Location/scale and empirical domain bounds use training observations only. Windows never cross cells; irregular cycle gaps are rejected. Anchors and targets retain physical percent units.

If cells share a pack, production batch, or test rig, split at that higher dependency unit before using this workflow. Distinct IDs alone do not guarantee experimental independence.

## Uncertainty

For each prediction, the nonconformity score is the maximum of lower-bound overshoot and upper-bound undershoot. Scores are maximized over horizons and windows **within each calibration cell**. The correction uses rank `ceil((n_cells + 1) * (1 - alpha))` with `alpha=0.2`. Too few independent cells cause an explicit error. Corrections are clamped to nonnegative values, conservatively widening intervals.

This is a grouped adaptation of [conformalized quantile regression](https://arxiv.org/abs/1905.03222). Nominal simultaneous coverage requires exchangeable independent cells and an equivalent window-sampling protocol. It is not guaranteed under domain shift. Reported pointwise coverage is distinct from simultaneous cell coverage.

The robust baseline uses observed slope spread and residual scale to create a **sensitivity envelope**. It is neither calibrated nor assigned nominal probabilistic coverage.

## Serving gates

Artifacts must declare validated status, measured provenance, disjoint partitions, architecture identity, and the matching weight checksum. Inference also checks feature order, chemistry, lookback, supported horizon, consecutive cycles, and empirical feature support. Checkpoints load on CPU with `weights_only=True` and strict state-dict matching. Only trusted local artifacts should be installed; checksums detect corruption, not malicious authorship.

Learned models currently extend observed operating conditions. Changed temperature/C-rate scenarios are rejected for learned models; use the explicit baseline for scenario sensitivity. Out-of-support histories are rejected rather than silently labeled in-domain.

RUL is the first EOL crossing within the supported horizon. No crossing returns `rul_cycles: null` and `rul_censored: true`. Already-degraded cells return zero. A short-horizon checkpoint never implies validated long-horizon life prediction.

## Empirical work still required

Train on audited measured observations, compare all models against persistence and robust trend, and report per-cell/horizon MAE/RMSE, interval coverage/width, and EOL-event/censor counts. Reserve an external cohort for final assessment. Repeated test-set tuning invalidates an untouched assessment. None of these empirical studies was performed during this modernization.

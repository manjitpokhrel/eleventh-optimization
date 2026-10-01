The Eleventh Optimization (safety-diff)
Safety alignment is evaluated on reference-precision models, but production serves compiled and quantized builds. Whether refusal behavior and its internal representation (the refusal direction, Arditi et al., 2024) survive that transformation is under-measured.

The Threat Model
If torch.compile (kernel fusion, CUDA graphs) or INT4 quantization (AWQ, GPTQ) silently shifts or destroys the refusal direction in the residual stream, models could pass standard fp16 safety evals but fail in production. This is a critical, unexamined deployment vulnerability.

The Pipeline
We are building safety-diff: a noise-controlled pipeline that tests whether safety alignment survives the inference stack.

Core Design Principles:

Noise-controlled: Every effect is measured against a run-to-run floating-point noise floor (reference vs. reference).
Rigorous Controls: Mandatory positive (INT3/weight noise) and specificity (perplexity/MMLU) controls.
Causal Validation: Layer-wise activation patching, directional ablation (necessity), and steering sweeps (sufficiency).
Multilingual Axis: Testing both English and Nepali to verify if safety degradation is language-agnostic.
Pre-registered Hypotheses
This work is strictly pre-registered before any evaluation runs.

H1: torch.compile stays within the noise floor.
H2: 4-bit builds (AWQ, GPTQ) exceed the noise floor on refusal rate or direction cosine.
H3: Nepali degradation ≥ English degradation.
H4: Safety change exceeds what general capability degradation predicts.

Kill Rule: If no build exceeds the noise floor, the deliverable becomes a rigorous negative-result study of stack invariance plus the safety-diff tool.

Status
Stage 1: Local debugging of activation hooks on Llama-3.2-1B. Rapid Grant submitted for 8B compute.

License
MIT

# Real GPU/LoRA canary

The real backend is intentionally separate from the CPU quickstart and from
CI. This is a manual operational check, not a benchmark or a release proof.

Current repository status: **PENDING**. No GPU/model-access run is recorded by
this change. The v3.2.0 release must remain blocked until an operator runs the
canary in an environment with CUDA, the base model, the checked-in adapter (or
an explicitly configured adapter), and any required model-provider access.

## Run

From the repository root, after installing the optional dependencies in
`requirements-gpu.txt`:

```bash
COMPLIANCE_GUARD_BACKEND=real \
COMPLIANCE_GUARD_RUN_GPU_CANARY=1 \
python3 scripts/run_gpu_canary.py
```

Optional configuration is supplied through environment variables:

- `COMPLIANCE_GUARD_BASE_MODEL` (default `mistralai/Mistral-7B-v0.1`)
- `COMPLIANCE_GUARD_LORA_ADAPTER` (default `models/checkpoints`)
- `COMPLIANCE_GUARD_TOKENIZER` (default adapter path when it exists)
- `COMPLIANCE_GUARD_ALLOW_CPU=1` only for a deliberate non-GPU diagnostic

Do not commit credentials, model tokens, generated outputs, or canary logs.
The script exits with status 2 when the guard is not explicitly enabled or
when CUDA/model access is unavailable. A successful run only demonstrates one
inference path; it does not establish compliance accuracy, safety, latency, or
production readiness.

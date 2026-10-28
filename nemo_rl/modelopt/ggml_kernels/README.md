# GGML IQ packers with importance weighting

A verbatim copy of ModelOpt's `modelopt/torch/kernels/quantization/ggml/` at commit
4cffe7f18ffe. The NeMo-RL image ships ModelOpt 0.48, whose GGML packers search plain squared
error; these take an optional per-value importance. `nemo_rl/modelopt/gguf_codes.py` builds them
as a separate extension (`weighted_ext`) for imatrix-weighted code training. Drop this copy once the
image's ModelOpt has `importance` on its packers.

Third-party attribution
=======================

`legacy/`: extracted from the user-provided GLM-5.3-Flash Kaggle notebook,
originally from https://github.com/ARahim3/kaggle-tpu-lab, MIT License.
See `kaggle-tpu-lab.LICENSE`. The original notebook is preserved as an input
artifact; its server and tunnel are not executed by the CPU workflow.

`src/qwen_tpu/jax_engine.py`: equations and checkpoint layouts ported from
the Qwen4Exp implementation in Hugging Face Transformers 5.19.0,
Copyright 2026 Qwen Team and Hugging Face, Apache License 2.0.
See `transformers.LICENSE` and
https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen4_exp.

`third_party/SpecForge/`: unmodified upstream source installed by the bootstrap,
https://github.com/sgl-project/SpecForge at commit
d80daf31aacfd2193283ae36f36864e4dde199bc. Its LICENSE and per-file notices
remain in that directory and in the generated notebook's embedded archive.

No pretrained model weights are included in this project. Generated tiny-model
weights under runs/ are random test fixtures, not Qwen pretrained checkpoints.

`third_party/Torchax/`: unmodified Google Torchax Python package, Apache-2.0,
https://github.com/google/torchax at commit
d95b0a5bff13e2b945e6a0e7ba3b3937b838161b. Its LICENSE and per-file notices
are preserved in the checkout and embedded notebook archive.

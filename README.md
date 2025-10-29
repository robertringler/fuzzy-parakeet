# Fuzzy Parakeet Superintelligence Toolkit

This repository provides a research-oriented PyTorch implementation of a
Transformer-style language model featuring rotary embeddings, root-mean-square
normalisation, and a sparse mixture-of-experts feed-forward network. The goal is
to offer a clean and easily-extensible codebase rather than claim any specific
capabilities.

## Getting started

```bash
python -m superintelligence.cli --steps 5 --sequence-length 128 --device cpu
```

This command will run a short synthetic training loop to verify the plumbing
between the model, data iterator, and trainer components.

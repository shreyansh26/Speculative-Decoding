# Speculative Decoding From Scratch

This repository implements speculative decoding methods behind one shared decoding contract:

- one target-model interface
- one greedy verifier
- one metrics schema
- one baseline autoregressive path

Execution order follows `plans/plan.md`:

1. Phase 0: common infrastructure
2. Phase 1: n-gram
3. Phase 2: draft model
4. Phase 3: Medusa-1
5. Phase 4: PARD
6. Phase 5: EAGLE-3
7. Phase 6: suffix decoding

The project uses `uv` with Python 3.12:

```bash
uv venv --python 3.12 .venv
uv pip install -e .
```

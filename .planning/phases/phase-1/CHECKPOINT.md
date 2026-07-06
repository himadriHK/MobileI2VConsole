# Checkpoint

**Saved:** 2026-07-02T10:43:00+05:30
**Phase:** 1
**Status:** complete
**Plan confirmed:** yes

## What was done

1. **Fixed Turbo-VAED decoder ONNX export bug** — `inject_noise` tuple length mismatch causing `IndexError: tuple index out of range`:
   - Root cause: `inject_noise` default tuple had 4 elements but needs 5 (accessed at indices 0..4 for mid block + 4 up blocks)
   - Fix: Changed both constructor default (line 585) and factory `setdefault` (line 783) from 4-element to 5-element tuple
   - Regression test: `scripts/convert/tests/test_turbo_vaed_bug.py` — validates instantiation succeeds with default args
   - Recorded as `F-1` in FAILURES.json

## What's next

Run `/fd-done` to close phase-1. The conversion scripts are ready for ONNX export execution on a GPU with sufficient VRAM.

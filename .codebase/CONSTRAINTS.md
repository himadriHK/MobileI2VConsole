# Architecture Constraints

## ONNX Export
- All model converters must run verify_onnx() before being considered complete
- ONNX export scripts use `torch.onnx.export()` (stable API), never `torch.onnx.dynamo_export`
- Model caches using device/dtype as keys must convert to str() first
- All SymInt values must be converted to int() before dict key lookup
- Dynamic control flow (while-loops on tensor values) is not allowed in ONNX-exported code paths

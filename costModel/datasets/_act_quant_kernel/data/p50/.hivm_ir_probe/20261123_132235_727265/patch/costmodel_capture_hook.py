
import json
import os
import re
import threading
from pathlib import Path

_LOCK = threading.Lock()
_COUNTER = 0

def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "kernel"

def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (bytes, bytearray, memoryview)):
        path.write_bytes(bytes(value))
    else:
        path.write_text(str(value), encoding="utf-8")

def _capture(jit_function, compiled):
    global _COUNTER
    output_dir = os.environ.get("COSTMODEL_CAPTURE_DIR")
    target = os.environ.get("COSTMODEL_CAPTURE_KERNEL", "")
    if not output_dir:
        return

    function_name = getattr(jit_function, "__name__", None)
    if not function_name:
        function = getattr(jit_function, "fn", None)
        function_name = getattr(function, "__name__", "")
    function_text = str(function_name)
    target_alt = target.lstrip("_")
    if target and target not in function_text and target_alt not in function_text:
        return

    asm = getattr(compiled, "asm", None)
    if not isinstance(asm, dict):
        return

    extensions = {
        "ttir": "ttir.mlir",
        "ttadapter": "ttadapter.mlir",
        "npuir": "npuir.mlir",
        "hivm": "module.hivm.mlir",
        "hivm_opt": "module.hivm.opt.mlir",
        "npubin": "npubin",
    }

    with _LOCK:
        _COUNTER += 1
        call_index = _COUNTER

    root = Path(output_dir)
    saved = []
    for key, value in asm.items():
        normalized = str(key).lower().replace("-", "_")
        extension = extensions.get(normalized)
        if extension is None:
            continue
        filename = "p%06d_k%03d_%s.%s" % (
            os.getpid(), call_index, _safe(function_name), extension
        )
        path = root / filename
        try:
            _write(path, value)
            saved.append(str(path))
        except Exception:
            pass

    if saved:
        event = {
            "pid": os.getpid(),
            "call_index": call_index,
            "jit_function": str(function_name),
            "asm_keys": [str(key) for key in asm.keys()],
            "saved_files": saved,
        }
        event_path = root.parent / "capture_events.jsonl"
        try:
            event_path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK:
                with event_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

try:
    from triton.runtime.jit import JITFunction
    _ORIGINAL_RUN = JITFunction.run

    def _costmodel_run(self, *args, **kwargs):
        compiled = _ORIGINAL_RUN(self, *args, **kwargs)
        try:
            _capture(self, compiled)
        except Exception:
            pass
        return compiled

    JITFunction.run = _costmodel_run
except Exception as exc:
    try:
        error_file = Path(os.environ.get("COSTMODEL_CAPTURE_DIR", ".")).parent / "capture_hook_error.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        error_file.write_text(repr(exc), encoding="utf-8")
    except Exception:
        pass

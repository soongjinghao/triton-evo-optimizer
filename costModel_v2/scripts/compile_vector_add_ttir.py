import os
import json
import time
import shutil
from pathlib import Path

# 必须在导入 Triton 之前设置
os.environ["TRITON_DEBUG"] = "1"
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_DISABLE_CACHE"] = "1"

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "ir" / "ttir" / "vector_add"

N_ELEMENTS = 4099
BLOCK_SIZES = [128, 256, 512]
DEVICE_INDEX = 0


def get_dump_roots():
    roots = []

    if os.environ.get("TRITON_DUMP_DIR"):
        roots.append(Path(os.environ["TRITON_DUMP_DIR"]))

    triton_home = Path(
        os.environ.get("TRITON_HOME", str(Path.home()))
    )
    roots.append(triton_home / ".triton" / "dump")
    roots.append(Path.home() / ".triton" / "dump")

    return list(dict.fromkeys(p.expanduser() for p in roots))


def find_ttir(start_ns):
    files = []

    for root in get_dump_roots():
        if not root.exists():
            continue

        for path in root.rglob("*.ttir.mlir"):
            try:
                stat = path.stat()

                if (
                    path.is_file()
                    and stat.st_size > 0
                    and stat.st_mtime_ns >= start_ns - 3_000_000_000
                ):
                    files.append(path)

            except OSError:
                continue

    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime_ns)


def main():
    print("=" * 70)
    print("Triton Vector Add TTIR Compilation")
    print("=" * 70)

    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable")

    torch.npu.set_device(DEVICE_INDEX)
    device = torch.device(f"npu:{DEVICE_INDEX}")

    x = torch.arange(
        N_ELEMENTS, device=device, dtype=torch.float32
    )
    y = torch.full(
        (N_ELEMENTS,), 2.0, device=device, dtype=torch.float32
    )
    output = torch.empty_like(x)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for block_size in BLOCK_SIZES:
        print()
        print("-" * 60)
        print(f"Compiling BLOCK_SIZE={block_size}")

        start_ns = time.time_ns()

        grid = (triton.cdiv(N_ELEMENTS, block_size),)

        vector_add_kernel[grid](
            x,
            y,
            output,
            N_ELEMENTS,
            BLOCK_SIZE=block_size,
        )

        torch.npu.synchronize()

        expected = x + y

        if not torch.equal(output, expected):
            raise RuntimeError(
                f"Correctness check failed: BLOCK_SIZE={block_size}"
            )

        ttir = find_ttir(start_ns)

        if ttir is None:
            raise RuntimeError(
                f"TTIR not found for BLOCK_SIZE={block_size}. "
                f"Checked: {get_dump_roots()}"
            )

        destination = OUTPUT_DIR / f"bs{block_size}.ttir.mlir"
        shutil.copy2(ttir, destination)

        metadata = {
            "kernel": "vector_add",
            "candidate_id": f"bs{block_size}",
            "block_size": block_size,
            "n_elements": N_ELEMENTS,
            "grid": list(grid),
            "device": f"npu:{DEVICE_INDEX}",
            "triton_version": triton.__version__,
            "status": "success",
            "ttir_path": str(destination.resolve()),
        }

        metadata_path = OUTPUT_DIR / f"bs{block_size}.metadata.json"

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        results.append(metadata)

        print("[SUCCESS] Compilation and correctness check passed")
        print("[TTIR]", destination)

    summary = {
        "kernel": "vector_add",
        "success_count": len(results),
        "results": results,
    }

    summary_path = OUTPUT_DIR / "compile_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("COMPILATION COMPLETED")
    print("Successful candidates:", len(results))
    print("Output directory:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()

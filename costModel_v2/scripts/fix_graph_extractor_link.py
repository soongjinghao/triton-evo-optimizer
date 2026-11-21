from pathlib import Path
import re
import shutil

ROOT = Path("/workspace/triton-ascend-src")

# 1. TritonNvidiaGPUTransforms 的 CMake 文件
nvidia_cmake = (
    ROOT / "lib/Dialect/TritonNvidiaGPU/Transforms/CMakeLists.txt"
)

# 2. 图提取器注册文件
extractor_cmake = (
    ROOT / "third_party/ascend/CMakeLists.txt"
)


def backup(path):
    bak = path.with_name(path.name + ".before_graph_link_fix")
    if not bak.exists():
        shutil.copy2(path, bak)
        print("[BACKUP]", bak)


# --------------------------------------------------
# Step 1: Add Utility.cpp to TritonNvidiaGPUTransforms
# --------------------------------------------------

backup(nvidia_cmake)

content = nvidia_cmake.read_text()

if not re.search(r"(?m)^\s*Utility\.cpp\s*$", content):
    pattern = r"(?m)^(\s*TMAUtilities\.cpp\s*)$"

    if not re.search(pattern, content):
        raise RuntimeError("Cannot locate TMAUtilities.cpp")

    content = re.sub(
        pattern,
        r"\1\n  Utility.cpp",
        content,
        count=1
    )

    nvidia_cmake.write_text(content)
    print("[PATCH] Added Utility.cpp to TritonNvidiaGPUTransforms")
else:
    print("[SKIP] Utility.cpp already exists")


# --------------------------------------------------
# Step 2: Add Transform targets to extractor
# --------------------------------------------------

backup(extractor_cmake)

content = extractor_cmake.read_text()

pattern = (
    r"(target_link_libraries"
    r"\(costmodel_v2_ttir_graph_extractor\s+PRIVATE)"
    r"(.*?)"
    r"(\))"
)

match = re.search(pattern, content, re.S)

if not match:
    raise RuntimeError("Cannot locate extractor target_link_libraries")

body = match.group(2)

required = [
    "TritonGPUTransforms",
    "TritonNvidiaGPUTransforms",
]

for target in required:
    if not re.search(
        rf"(?m)^\s*{re.escape(target)}\s*$",
        body
    ):
        body = body.rstrip() + "\n            " + target + "\n"
        print("[PATCH] Added link dependency:", target)
    else:
        print("[SKIP] Already linked:", target)

content = (
    content[:match.start()]
    + match.group(1)
    + body
    + match.group(3)
    + content[match.end():]
)

extractor_cmake.write_text(content)

print()
print("CMake modifications completed.")

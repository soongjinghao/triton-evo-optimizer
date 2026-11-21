from pathlib import Path
import re
import shutil

ROOT = Path("/workspace/triton-ascend-src")

GPU_CMAKE = ROOT / "lib/Dialect/TritonGPU/CMakeLists.txt"
GPU_TRANSFORMS = ROOT / "lib/Dialect/TritonGPU/Transforms/CMakeLists.txt"
NVIDIA_TRANSFORMS = ROOT / "lib/Dialect/TritonNvidiaGPU/Transforms/CMakeLists.txt"
ASCEND_CMAKE = ROOT / "third_party/ascend/CMakeLists.txt"

NVWS_INCLUDE = ROOT / "third_party/nvidia/include/Dialect/NVWS"
NVWS_LIB = ROOT / "third_party/nvidia/lib/Dialect/NVWS"

files = [
    GPU_CMAKE, GPU_TRANSFORMS,
    NVIDIA_TRANSFORMS, ASCEND_CMAKE
]

for p in files:
    if not p.is_file():
        raise FileNotFoundError(p)

for p in [
    NVWS_INCLUDE / "CMakeLists.txt",
    NVWS_LIB / "CMakeLists.txt",
    ROOT / "lib/Dialect/TritonGPU/Transforms/Utility.cpp",
    ROOT / "lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp",
]:
    if not p.is_file():
        raise FileNotFoundError(p)

def backup(p):
    bak = Path(str(p) + ".before_costmodel_graph_fix")
    if not bak.exists():
        shutil.copy2(p, bak)
        print("[BACKUP]", bak)

for p in files:
    backup(p)

# --------------------------------------------------
# 1. Register only NVIDIA NVWS headers and libraries.
#    The order is important:
#    TritonGPUIR -> NVWSIR/NVWSTransforms -> TritonGPUTransforms
# --------------------------------------------------

text = GPU_CMAKE.read_text()

marker = "# BEGIN COSTMODEL_V2_NVWS_GRAPH_DEPS"

if marker not in text:
    lines = text.splitlines()

    ir_index = next(
        (i for i, line in enumerate(lines)
         if re.fullmatch(r"\s*add_subdirectory\(IR\)\s*", line)),
        None
    )

    transforms_index = next(
        (i for i, line in enumerate(lines)
         if re.fullmatch(r"\s*add_subdirectory\(Transforms\)\s*", line)),
        None
    )

    if ir_index is None or transforms_index is None or ir_index >= transforms_index:
        raise RuntimeError(
            "Unexpected TritonGPU/CMakeLists.txt structure; no patch applied."
        )

    block = r'''
# BEGIN COSTMODEL_V2_NVWS_GRAPH_DEPS

# NVWS generated headers and TableGen targets.
include_directories("${PROJECT_SOURCE_DIR}/third_party/nvidia/include")
include_directories("${PROJECT_BINARY_DIR}/third_party/nvidia/include")

if(NOT TARGET NVWSTableGen)
  add_subdirectory(
    "${PROJECT_SOURCE_DIR}/third_party/nvidia/include/Dialect/NVWS"
    "${PROJECT_BINARY_DIR}/third_party/nvidia/include/Dialect/NVWS"
  )
endif()

# NVWS IR and Transform libraries.
if(NOT TARGET NVWSIR)
  add_subdirectory(
    "${PROJECT_SOURCE_DIR}/third_party/nvidia/lib/Dialect/NVWS"
    "${PROJECT_BINARY_DIR}/third_party/nvidia/lib/Dialect/NVWS"
  )
endif()

# END COSTMODEL_V2_NVWS_GRAPH_DEPS
'''.strip("\n").splitlines()

    lines[ir_index + 1:ir_index + 1] = block
    GPU_CMAKE.write_text("\n".join(lines) + "\n")
    print("[PATCH] Registered NVWS before TritonGPUTransforms")
else:
    print("[SKIP] NVWS registration already exists")

# --------------------------------------------------
# 2. Check that the two required implementation files
#    are already registered in their respective targets.
# --------------------------------------------------

gpu_text = GPU_TRANSFORMS.read_text()
nvidia_text = NVIDIA_TRANSFORMS.read_text()

if not re.search(r"(?m)^\s*Utility\.cpp\s*$", gpu_text):
    raise RuntimeError("TritonGPUTransforms does not list Utility.cpp")

if not re.search(r"(?m)^\s*TMAUtilities\.cpp\s*$", nvidia_text):
    raise RuntimeError("TritonNvidiaGPUTransforms does not list TMAUtilities.cpp")

# Remove the previous experimental nested Utility.cpp entry.
# The required inferDstEncoding and verifyBarrierType implementations
# are already in the outer TritonGPU/Transforms/Utility.cpp.
nvidia_text_new, removed = re.subn(
    r'(?m)^\s*(?:\$\{PROJECT_SOURCE_DIR\}/)?'
    r'third_party/ascend/AscendNPU-IR/bishengir/triton/'
    r'lib/Dialect/TritonNvidiaGPU/Transforms/Utility\.cpp\s*\n',
    '',
    nvidia_text
)

if removed:
    NVIDIA_TRANSFORMS.write_text(nvidia_text_new)
    print("[PATCH] Removed redundant nested Utility.cpp entry")
else:
    print("[SKIP] No experimental nested Utility.cpp entry")

# --------------------------------------------------
# 3. Ensure the standalone extractor links both
#    transform targets and explicitly builds them.
# --------------------------------------------------

ascend_text = ASCEND_CMAKE.read_text()

pattern = re.compile(
    r'(target_link_libraries\s*\(\s*'
    r'costmodel_v2_ttir_graph_extractor\s+PRIVATE)'
    r'(.*?)'
    r'(\))',
    re.S
)

match = pattern.search(ascend_text)

if not match:
    raise RuntimeError("Cannot find extractor target_link_libraries block")

body = match.group(2)

for target in ["TritonGPUTransforms", "TritonNvidiaGPUTransforms"]:
    if not re.search(rf"\b{target}\b", body):
        body += f"\n    {target}"
        print("[PATCH] Added link dependency:", target)

ascend_text = (
    ascend_text[:match.start(2)]
    + body
    + ascend_text[match.end(2):]
)

dep_pattern = re.compile(
    r'add_dependencies\s*\(\s*'
    r'costmodel_v2_ttir_graph_extractor'
    r'([^)]*)\)'
)

dep_match = dep_pattern.search(ascend_text)

if dep_match:
    deps = dep_match.group(1).split()

    for target in ["TritonGPUTransforms", "TritonNvidiaGPUTransforms"]:
        if target not in deps:
            deps.append(target)

    replacement = (
        "add_dependencies(costmodel_v2_ttir_graph_extractor\n    "
        + "\n    ".join(deps)
        + "\n)"
    )

    ascend_text = (
        ascend_text[:dep_match.start()]
        + replacement
        + ascend_text[dep_match.end():]
    )
else:
    ascend_text += """
add_dependencies(costmodel_v2_ttir_graph_extractor
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
"""

ASCEND_CMAKE.write_text(ascend_text)

print("\n[OK] CMake repair patches completed.")

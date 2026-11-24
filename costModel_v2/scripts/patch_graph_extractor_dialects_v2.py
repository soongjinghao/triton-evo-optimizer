from pathlib import Path
import shutil

ROOT = Path("/workspace/triton-ascend-src")
MAIN = Path("/workspace/Agent/costModel_v2/cpp/ttir_graph_extractor/src/main.cpp")
CMAKE = ROOT / "third_party/ascend/CMakeLists.txt"

def backup(path, suffix):
    bak = Path(str(path) + suffix)
    if not bak.exists():
        shutil.copy2(path, bak)
        print("[BACKUP]", bak)

if not MAIN.exists() or not CMAKE.exists():
    raise RuntimeError("Source file not found")

s = MAIN.read_text()
c = CMAKE.read_text()

# ==========================================
# 1. Add MLIR and BiShengIR headers
# ==========================================

headers = [
    '#include "mlir/Dialect/Affine/IR/AffineOps.h"',
    '#include "mlir/Dialect/Func/IR/FuncOps.h"',
    '#include "mlir/Dialect/Linalg/IR/Linalg.h"',
    '#include "mlir/Dialect/MemRef/IR/MemRef.h"',
    '#include "mlir/Dialect/Vector/IR/VectorOps.h"',
    '#include "bishengir/Dialect/Annotation/IR/Annotation.h"',
    '#include "bishengir/Dialect/HACC/IR/HACC.h"',
    '#include "bishengir/Dialect/HFusion/IR/HFusion.h"',
    '#include "bishengir/Dialect/HIVM/IR/HIVM.h"',
]

anchor = '#include "mlir/Dialect/Arith/IR/Arith.h"\n'

if anchor not in s:
    raise RuntimeError("Cannot find Arith include anchor")

missing = [h for h in headers if h not in s]

if missing:
    s = s.replace(
        anchor,
        anchor + "\n".join(missing) + "\n",
        1
    )

# ==========================================
# 2. Register additional dialects
# ==========================================

if "mlir::hivm::HIVMDialect" not in s:

    old = "      mlir::gpu::GPUDialect>();"

    new = """      mlir::gpu::GPUDialect,
      mlir::func::FuncDialect,
      mlir::affine::AffineDialect,
      mlir::linalg::LinalgDialect,
      mlir::memref::MemRefDialect,
      mlir::vector::VectorDialect,
      mlir::annotation::AnnotationDialect,
      mlir::hacc::HACCDialect,
      mlir::hfusion::HFusionDialect,
      mlir::hivm::HIVMDialect>();"""

    if old not in s:
        raise RuntimeError("Cannot find dialect registration anchor")

    s = s.replace(old, new, 1)

# ==========================================
# 3. Add ParseOnly option
# ==========================================

if "static llvm::cl::opt<bool> ParseOnly(" not in s:

    anchor = "static std::string sanitizeName("

    if anchor not in s:
        raise RuntimeError("Cannot find sanitizeName anchor")

    option = '''static llvm::cl::opt<bool> ParseOnly(
    "parse-only",
    llvm::cl::desc("Only parse MLIR without graph extraction"),
    llvm::cl::init(false));

'''

    s = s.replace(anchor, option + anchor, 1)

# ==========================================
# 4. Add ParseOnly branch
# ==========================================

if "if (ParseOnly)" not in s:

    anchor = "  // 3. Create output directory."

    if anchor not in s:
        raise RuntimeError("Cannot find output directory anchor")

    branch = '''  if (ParseOnly) {
    llvm::errs() << "[PASS] MLIR parsing completed\\\\n";
    return 0;
  }

'''

    s = s.replace(anchor, branch + anchor, 1)

# ==========================================
# 5. Add CMake link libraries
# ==========================================

target = "target_link_libraries(costmodel_v2_ttir_graph_extractor PRIVATE"

start = c.find(target)

if start < 0:
    raise RuntimeError("Cannot find extractor link target")

depth = 0
end = -1

for i in range(start, len(c)):
    if c[i] == "(":
        depth += 1
    elif c[i] == ")":
        depth -= 1
        if depth == 0:
            end = i
            break

if end < 0:
    raise RuntimeError("Cannot parse CMake link block")

block = c[start:end + 1]
lines = block.splitlines()

anchor_index = next(
    (i for i, line in enumerate(lines)
     if line.strip() == "MLIRGPUDialect"),
    None
)

if anchor_index is None:
    raise RuntimeError("Cannot find MLIRGPUDialect in CMake")

libraries = [
    "MLIRAffineDialect",
    "MLIRLinalgDialect",
    "MLIRMemRefDialect",
    "MLIRVectorDialect",
    "BiShengIRAnnotationDialect",
    "BiShengIRHACCDialect",
    "BiShengIRHFusionDialect",
    "BiShengIRHIVMDialect",
]

existing = {line.strip() for line in lines}

missing_libs = [
    lib for lib in libraries if lib not in existing
]

lines[anchor_index + 1:anchor_index + 1] = [
    "            " + lib for lib in missing_libs
]

new_block = "\n".join(lines)
c = c[:start] + new_block + c[end + 1:]

# ==========================================
# 6. Validate before writing
# ==========================================

required = [
    "mlir::func::FuncDialect",
    "mlir::annotation::AnnotationDialect",
    "mlir::hacc::HACCDialect",
    "mlir::hfusion::HFusionDialect",
    "mlir::hivm::HIVMDialect",
    "if (ParseOnly)",
]

for item in required:
    if item not in s:
        raise RuntimeError("Validation failed: " + item)

for lib in libraries:
    if lib not in c:
        raise RuntimeError("CMake validation failed: " + lib)

# ==========================================
# 7. Backup and write
# ==========================================

backup(MAIN, ".before_graph_dialects_v2")
backup(CMAKE, ".before_graph_dialects_v2")

MAIN.write_text(s)
CMAKE.write_text(c)

print("\n[SUCCESS] main.cpp updated")
print("[SUCCESS] CMakeLists.txt updated")
print("[SUCCESS] Dialect registration completed")
print("[SUCCESS] ParseOnly option added")
print("[SUCCESS] All validations passed")

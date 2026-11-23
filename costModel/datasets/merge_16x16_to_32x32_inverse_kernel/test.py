import torch
import numpy as np
import triton
from typing import Optional

from merge_16x16_to_32x32_inverse_kernel import solve_tril

def test_merge_16x16_to_32x32_inverse():
    """Test for solve_tril with BT=32."""
    
    # Test parameters
    B, T, H, BT = 2, 64, 4, 32  # Smaller values for easier debugging
    device = torch.device('npu')
    
    print(f"Testing merge_16x16_to_32x32_inverse (corrected) with B={B}, T={T}, H={H}, BT={BT}")
    print(f"Using device: {device}")
    
    # Number of 32x32 blocks per sequence
    num_blocks = T // BT  # 64 / 32 = 2 blocks per sequence
    
    # Create input tensor A with shape [B, T, H, 32]
    # IMPORTANT: The kernel only reads specific parts of this tensor:
    # - For rows 0-15: only columns 0-15 are read (L₁₁)
    # - For rows 16-31: columns 0-15 are read (L₂₁) and columns 16-31 are read (L₂₂)
    # The rest of the tensor (rows 0-15, cols 16-31) can be arbitrary - kernel doesn't read them!
    A = torch.randn(B, T, H, BT, device=device, dtype=torch.float32)
    
    print("\nGenerating strictly lower triangular blocks...")
    
    # Make A strictly lower triangular in the blocks that matter
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                start_row = block_idx * BT
                
                # Create strictly lower triangular L₁₁ (rows 0-15, cols 0-15)
                for i in range(16):
                    for j in range(i, 16):  # j >= i (including diagonal)
                        A[b, start_row + i, h, j] = 0.0
                
                # Create strictly lower triangular L₂₁ (rows 16-31, cols 0-15)
                for i in range(16, 32):
                    for j in range(i - 16, 16):  # Adjusted indices for 16x16 block
                        A[b, start_row + i, h, j] = 0.0
                
                # Create strictly lower triangular L₂₂ (rows 16-31, cols 16-31)
                for i in range(16, 32):
                    for j in range(16 + (i - 16), 32):  # j >= i in the 16×16 block
                        A[b, start_row + i, h, j] = 0.0
    
    print(f"Created input tensor A with shape: {A.shape}")
    print(f"A dtype: {A.dtype}, device: {A.device}")
    
    # Call the solve_tril function
    Ai = solve_tril(A, output_dtype=torch.float32)
    
    print(f"Output Ai shape: {Ai.shape}")
    print(f"Ai dtype: {Ai.dtype}")
    
    # Verify the output shape
    assert Ai.shape == (B, T, H, BT), f"Expected shape {(B, T, H, BT)}, got {Ai.shape}"
    
    # Convert to numpy for verification
    A_np = A.cpu().numpy()
    Ai_np = Ai.cpu().numpy()
    
    print("\n=== Test 1: Block-wise formula verification ===")
    
    max_formula_error = 0.0
    blocks_tested = 0
    
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                start_row = block_idx * BT
                
                # Extract the subblocks that the kernel actually reads
                # Note: The kernel reads these from input A
                L11 = np.zeros((16, 16), dtype=np.float32)  # Actually L₁₁
                L21 = np.zeros((16, 16), dtype=np.float32)  # Actually L₂₁  
                L22 = np.zeros((16, 16), dtype=np.float32)  # Actually L₂₂
                
                # Extract L₁₁ from A (rows 0-15, cols 0-15)
                for i in range(16):
                    for j in range(16):
                        L11[i, j] = A_np[b, start_row + i, h, j]
                
                # Extract L₂₁ from A (rows 16-31, cols 0-15)
                for i in range(16, 32):
                    for j in range(16):
                        L21[i-16, j] = A_np[b, start_row + i, h, j]
                
                # Extract L₂₂ from A (rows 16-31, cols 16-31)
                for i in range(16, 32):
                    for j in range(16, 32):
                        L22[i-16, j-16] = A_np[b, start_row + i, h, j]
                
                # Extract the computed inverse blocks from Ai
                Ai11 = np.zeros((16, 16), dtype=np.float32)  # Should be (I+L₁₁)^(-1)
                Ai21 = np.zeros((16, 16), dtype=np.float32)  # Should be -(I+L₂₂)^(-1)*L₂₁*(I+L₁₁)^(-1)
                Ai22 = np.zeros((16, 16), dtype=np.float32)  # Should be (I+L₂₂)^(-1)
                
                # Extract Ai11 from output (rows 0-15, cols 0-15)
                for i in range(16):
                    for j in range(16):
                        Ai11[i, j] = Ai_np[b, start_row + i, h, j]
                
                # Extract Ai21 from output (rows 16-31, cols 0-15)
                for i in range(16, 32):
                    for j in range(16):
                        Ai21[i-16, j] = Ai_np[b, start_row + i, h, j]
                
                # Extract Ai22 from output (rows 16-31, cols 16-31)
                for i in range(16, 32):
                    for j in range(16, 32):
                        Ai22[i-16, j-16] = Ai_np[b, start_row + i, h, j]
                
                # I16 identity matrix
                I16 = np.eye(16, dtype=np.float32)
                
                # Check 1: Ai11 should be inverse of (I + L₁₁)
                # Ai11 * (I + L₁₁) ≈ I
                error11 = np.max(np.abs(np.matmul(Ai11, I16 + L11) - I16))
                
                # Check 2: Ai22 should be inverse of (I + L₂₂)
                # Ai22 * (I + L₂₂) ≈ I
                error22 = np.max(np.abs(np.matmul(Ai22, I16 + L22) - I16))
                
                # Check 3: Ai21 should satisfy the formula
                # Ai21 ≈ -Ai22 * L₂₁ * Ai11
                expected_Ai21 = -np.matmul(np.matmul(Ai22, L21), Ai11)
                error21 = np.max(np.abs(Ai21 - expected_Ai21))
                
                # Check 4: Upper-right block of output should be zero
                # Extract Ai12 (rows 0-15, cols 16-31) from output
                Ai12 = np.zeros((16, 16), dtype=np.float32)
                for i in range(16):
                    for j in range(16, 32):
                        Ai12[i, j-16] = Ai_np[b, start_row + i, h, j]
                error12 = np.max(np.abs(Ai12))
                
                block_error = max(error11, error22, error21, error12)
                max_formula_error = max(max_formula_error, block_error)
                
                if block_error > 1e-3:
                    print(f"  Block [{b}, {h}, {block_idx}] formula errors:")
                    print(f"    Ai11*(I+L11)-I: {error11:.2e}")
                    print(f"    Ai22*(I+L22)-I: {error22:.2e}")
                    print(f"    Ai21 + Ai22*L21*Ai11: {error21:.2e}")
                    print(f"    Upper-right zero: {error12:.2e}")
                
                blocks_tested += 1
    
    print(f"\nFormula verification completed:")
    print(f"  Blocks tested: {blocks_tested}")
    print(f"  Maximum formula error: {max_formula_error:.2e}")
    
    if max_formula_error < 1e-3:
        print("✓ All block formulas correct within 1e-3")
    else:
        print(f"✗ Formula errors detected")
    
    print("\n=== Test 2: Full 32×32 block verification ===")
    
    max_block_error = 0.0
    
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                start_row = block_idx * BT
                
                # Reconstruct the ACTUAL 32×32 strictly lower triangular matrix
                # that corresponds to what the kernel computes the inverse of
                A_block_actual = np.zeros((32, 32), dtype=np.float32)
                
                # Fill L₁₁ (rows 0-15, cols 0-15) - strictly lower triangular
                for i in range(16):
                    for j in range(16):
                        if j < i:  # Strictly lower triangular
                            A_block_actual[i, j] = A_np[b, start_row + i, h, j]
                        # else: diagonal and above are zero (already initialized)
                
                # Fill L₂₁ (rows 16-31, cols 0-15) - strictly lower triangular
                for i in range(16, 32):
                    for j in range(16):
                        if j < (i - 16):  # Strictly lower triangular in 16×16 block
                            A_block_actual[i, j] = A_np[b, start_row + i, h, j]
                
                # Fill L₂₂ (rows 16-31, cols 16-31) - strictly lower triangular
                for i in range(16, 32):
                    for j in range(16, 32):
                        if j < i:  # Strictly lower triangular in full 32×32
                            A_block_actual[i, j] = A_np[b, start_row + i, h, j]
                
                # Reconstruct the computed inverse from output
                Ai_block = np.zeros((32, 32), dtype=np.float32)
                
                # Fill Ai11 (rows 0-15, cols 0-15)
                for i in range(16):
                    for j in range(16):
                        Ai_block[i, j] = Ai_np[b, start_row + i, h, j]
                
                # Fill Ai21 (rows 16-31, cols 0-15)
                for i in range(16, 32):
                    for j in range(16):
                        Ai_block[i, j] = Ai_np[b, start_row + i, h, j]
                
                # Fill Ai22 (rows 16-31, cols 16-31)
                for i in range(16, 32):
                    for j in range(16, 32):
                        Ai_block[i, j] = Ai_np[b, start_row + i, h, j]
                
                # Upper-right (rows 0-15, cols 16-31) remains zero
                
                # Compute (I + A_block_actual) * Ai_block
                I32 = np.eye(32, dtype=np.float32)
                product = np.matmul(I32 + A_block_actual, Ai_block)
                
                # Should be identity
                error = np.max(np.abs(product - I32))
                max_block_error = max(max_block_error, error)
                
                if error > 1e-3:
                    print(f"  Block [{b}, {h}, {block_idx}] full multiplication error: {error:.2e}")
                    # Debug: print the largest element-wise error
                    diff = np.abs(product - I32)
                    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                    print(f"    Max error at ({max_idx[0]}, {max_idx[1]}): {diff[max_idx]:.2e}")
    
    print(f"\nFull block verification completed:")
    print(f"  Blocks tested: {blocks_tested}")
    print(f"  Maximum block error: {max_block_error:.2e}")
    
    if max_block_error < 1e-3:
        print("✓ All 32×32 blocks correct within 1e-3")
    else:
        print(f"✗ Block errors detected")
    
    print("\n=== Test 3: Structure verification ===")
    
    max_structure_error = 0.0
    
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                start_row = block_idx * BT
                
                # Check diagonal ≈ 1
                for i in range(32):
                    if start_row + i < T:
                        diag_val = Ai_np[b, start_row + i, h, i]
                        max_structure_error = max(max_structure_error, abs(diag_val - 1.0))
                
                # Check upper-right quadrant is zero
                for i in range(16):
                    for j in range(16, 32):
                        if start_row + i < T:
                            val = Ai_np[b, start_row + i, h, j]
                            max_structure_error = max(max_structure_error, abs(val))
    
    print(f"Maximum structure error: {max_structure_error:.2e}")
    
    if max_structure_error < 1e-3:
        print("✓ Inverse has correct structure")
    else:
        print(f"✗ Structure issues detected")
    
    # Final checks
    print("\n" + "="*60)
    if max_formula_error < 1e-3 and max_block_error < 1e-3 and max_structure_error < 1e-3:
        print("SUCCESS: All tests passed!")
        return A, Ai
    else:
        raise AssertionError(
            f"Tests failed:\n"
            f"  Formula error: {max_formula_error:.2e}\n"
            f"  Block error: {max_block_error:.2e}\n"
            f"  Structure error: {max_structure_error:.2e}"
        )

# Run the test
if __name__ == "__main__":
    try:
        A, Ai = test_merge_16x16_to_32x32_inverse()
        print("\n" + "="*60)
        print("SUCCESS: Corrected test passed!")
        print("="*60)
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

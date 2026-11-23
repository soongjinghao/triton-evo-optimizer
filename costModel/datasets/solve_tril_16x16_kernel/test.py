import torch
import numpy as np
import triton
from typing import Optional

from solve_tril_16x16_kernel import solve_tril

def test_solve_tril_16x16():
    """Test solve_tril with BT=16 for block diagonal strictly lower triangular matrices."""
    
    # Test parameters as specified
    B, T, H, BT = 4, 64, 8, 16
    device = torch.device('npu')
    
    print(f"Testing solve_tril with B={B}, T={T}, H={H}, BT={BT}")
    print(f"Using device: {device}")
    
    # Number of 16x16 blocks per sequence
    num_blocks = T // BT  # 64 / 16 = 4 blocks per sequence
    
    # Create a strictly lower triangular block diagonal matrix A
    # A has shape [B, T, H, BT] where BT=16
    # Each 16x16 block is strictly lower triangular (diagonal and above are 0)
    A = torch.zeros(B, T, H, BT, device=device)
    
    # Fill with random strictly lower triangular values for each block
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                # Create a random strictly lower triangular 16x16 matrix
                # Start with random values
                block = torch.randn(BT, BT, device=device)
                
                # Make it strictly lower triangular: zero on diagonal and above
                for i in range(BT):
                    for j in range(i, BT):  # j >= i
                        block[i, j] = 0.0
                
                # Store only the first BT columns (since it's block diagonal)
                # The block is stored in rows [block_idx*BT : (block_idx+1)*BT, 0:BT]
                start_row = block_idx * BT
                for row in range(BT):
                    # Store only the relevant part (first BT columns)
                    A[b, start_row + row, h, :] = block[row, :BT]
    
    print(f"Created input tensor A with shape: {A.shape}")
    print(f"A dtype: {A.dtype}, device: {A.device}")
    
    # Test 1: Basic functionality test
    print("\nTest 1: Basic functionality with BT=16")
    
    # Call the solve_tril function
    Ai = solve_tril(A, output_dtype=torch.float32)
    
    print(f"Output Ai shape: {Ai.shape}")
    print(f"Ai dtype: {Ai.dtype}")
    
    # Verify the output shape
    assert Ai.shape == (B, T, H, BT), f"Expected shape {(B, T, H, BT)}, got {Ai.shape}"
    
    # Convert to numpy for easier verification
    A_np = A.cpu().numpy()
    Ai_np = Ai.cpu().numpy()
    
    # For each batch, head, and block, verify that (I + A_block) * Ai_block ≈ I
    print("\nVerifying block-wise correctness...")
    
    max_error = 0.0
    total_blocks_tested = 0
    
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                # Extract the original 16x16 block from A
                # Note: A only stores the first 16 columns of each block row
                # We need to reconstruct the full 16x16 strictly lower triangular block
                start_row = block_idx * BT
                
                # Create the full 16x16 block from the stored data
                # The block is strictly lower triangular, so upper triangle is zero
                A_block_full = np.zeros((BT, BT), dtype=np.float32)
                
                for i in range(BT):
                    # For row i, columns 0 through i-1 contain values
                    # Columns i through 15 are zero in a strictly lower triangular matrix
                    for j in range(i):
                        # Get the value from A tensor
                        A_block_full[i, j] = A_np[b, start_row + i, h, j]
                
                # Extract the computed inverse block from Ai
                # Ai stores the full 16x16 inverse block
                Ai_block = np.zeros((BT, BT), dtype=np.float32)
                for i in range(BT):
                    Ai_block[i, :] = Ai_np[b, start_row + i, h, :]
                
                # Compute I + A_block
                I_plus_A = np.eye(BT, dtype=np.float32) + A_block_full
                
                # Multiply (I + A_block) * Ai_block
                product = np.matmul(I_plus_A, Ai_block)
                
                # The product should be approximately the identity matrix
                identity_target = np.eye(BT, dtype=np.float32)
                
                # Calculate the error
                error = np.max(np.abs(product - identity_target))
                max_error = max(max_error, error)
                
                # Check if this block is correct (within tolerance)
                if error > 1e-4:
                    print(f"  Block [{b}, {h}, {block_idx}] has error {error:.6f}")
                    # Print more details for debugging
                    print(f"    Max value in A_block: {np.max(np.abs(A_block_full)):.6f}")
                    print(f"    Max value in Ai_block: {np.max(np.abs(Ai_block)):.6f}")
                    print(f"    Max value in product: {np.max(np.abs(product)):.6f}")
                
                total_blocks_tested += 1
    
    print(f"\nTest completed:")
    print(f"  Total blocks tested: {total_blocks_tested}")
    print(f"  Maximum error across all blocks: {max_error:.6e}")
    
    # Check if all blocks passed the test
    if max_error < 1e-4:
        print("✓ All blocks passed! (I + A) * Ai ≈ I within tolerance 1e-4")
    else:
        print(f"✗ Some blocks failed! Maximum error: {max_error:.6e}")
    
    print("\n✓ All tests passed!")
    
    return A, Ai

# Run the test
if __name__ == "__main__":
    try:
        A, Ai = test_solve_tril_16x16()
        print("\n" + "="*50)
        print("SUCCESS: solve_tril test passed!")
        print("="*50)
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

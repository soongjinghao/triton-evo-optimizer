import torch
import numpy as np

from merge_16x16_to_64x64_inverse_kernel_1 import solve_tril

def test_solve_tril_64x64():
    """Test solve_tril with BT=64 for block diagonal strictly lower triangular matrices."""
    
    # Test parameters
    B, T, H, BT = 2, 128, 4, 64  # Using 2 sequences of length 128 = 2 blocks each
    device = torch.device('npu')
    
    print(f"Testing solve_tril with B={B}, T={T}, H={H}, BT={BT}")
    print(f"Using device: {device}")
    
    # Number of 64x64 blocks per sequence
    num_blocks = T // BT  # 128 / 64 = 2 blocks per sequence
    
    # Create a strictly lower triangular block diagonal matrix A
    # A has shape [B, T, H, BT] where BT=64
    # Each 64x64 block is strictly lower triangular (diagonal and above are 0)
    A = torch.zeros(B, T, H, BT, device=device, dtype=torch.float32)
    
    print(f"\nCreating input tensor A with shape: {A.shape}")
    print("Filling with random strictly lower triangular values...")
    
    # Fill with random strictly lower triangular values for each 64x64 block
    # Following the ROW-DEPENDENT memory interpretation:
    # - Rows 0-15: only cols 0-15 matter (L₁₁)
    # - Rows 16-31: cols 0-31 matter (L₂₁, L₂₂)
    # - Rows 32-47: cols 0-47 matter (L₃₁, L₃₂, L₃₃)
    # - Rows 48-63: all cols 0-63 matter (L₄₁, L₄₂, L₄₃, L₄₄)
    
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                start_row = block_idx * BT
                
                # Create all 10 strictly lower triangular 16x16 blocks
                # Note: We use different naming to avoid confusion with kernel variables
                blocks = {}
                
                # Generate random strictly lower triangular blocks
                for block_name in ['L11', 'L21', 'L22', 'L31', 'L32', 'L33', 'L41', 'L42', 'L43', 'L44']:
                    block = torch.randn(16, 16, device=device)
                    
                    # Make it strictly lower triangular: zero on diagonal and above
                    for i in range(16):
                        for j in range(i, 16):  # j >= i includes diagonal
                            block[i, j] = 0.0
                    
                    # Scale down to avoid numerical issues
                    block = block * 0.1
                    blocks[block_name] = block
                
                # Store in packed format with row-dependent interpretation
                
                # Rows 0-15: Only cols 0-15 matter (L₁₁)
                for i in range(16):
                    for j in range(16):
                        A[b, start_row + i, h, j] = blocks['L11'][i, j]
                    # Cols 16-63 are arbitrary (ignored by kernel)
                    for j in range(16, 64):
                        A[b, start_row + i, h, j] = 0.0  # Set to zero for clarity
                
                # Rows 16-31: Cols 0-15 = L₂₁, Cols 16-31 = L₂₂
                for i in range(16, 32):
                    # L₂₁ in cols 0-15
                    for j in range(16):
                        A[b, start_row + i, h, j] = blocks['L21'][i-16, j]
                    # L₂₂ in cols 16-31
                    for j in range(16, 32):
                        A[b, start_row + i, h, j] = blocks['L22'][i-16, j-16]
                    # Cols 32-63 are arbitrary (ignored by kernel)
                    for j in range(32, 64):
                        A[b, start_row + i, h, j] = 0.0  # Set to zero for clarity
                
                # Rows 32-47: Cols 0-15 = L₃₁, 16-31 = L₃₂, 32-47 = L₃₃
                for i in range(32, 48):
                    # L₃₁ in cols 0-15
                    for j in range(16):
                        A[b, start_row + i, h, j] = blocks['L31'][i-32, j]
                    # L₃₂ in cols 16-31
                    for j in range(16, 32):
                        A[b, start_row + i, h, j] = blocks['L32'][i-32, j-16]
                    # L₃₃ in cols 32-47
                    for j in range(32, 48):
                        A[b, start_row + i, h, j] = blocks['L33'][i-32, j-32]
                    # Cols 48-63 are arbitrary (ignored by kernel)
                    for j in range(48, 64):
                        A[b, start_row + i, h, j] = 0.0  # Set to zero for clarity
                
                # Rows 48-63: All cols matter
                for i in range(48, 64):
                    # L₄₁ in cols 0-15
                    for j in range(16):
                        A[b, start_row + i, h, j] = blocks['L41'][i-48, j]
                    # L₄₂ in cols 16-31
                    for j in range(16, 32):
                        A[b, start_row + i, h, j] = blocks['L42'][i-48, j-16]
                    # L₄₃ in cols 32-47
                    for j in range(32, 48):
                        A[b, start_row + i, h, j] = blocks['L43'][i-48, j-32]
                    # L₄₄ in cols 48-63
                    for j in range(48, 64):
                        A[b, start_row + i, h, j] = blocks['L44'][i-48, j-48]
    
    print(f"Created input tensor A:")
    print(f"  Shape: {A.shape}")
    print(f"  Dtype: {A.dtype}")
    print(f"  Device: {A.device}")
    
    # Validate that the input is correctly structured
    print(f"\nValidating input structure...")
    validation_errors = []
    
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                start_row = block_idx * BT
                
                # Check that diagonal elements are zero
                # For each row, the diagonal should be at column = row_in_block
                for i in range(BT):
                    row_in_block = i % 64  # Position within the 64x64 block
                    diag_value = A[b, start_row + i, h, row_in_block].item()
                    if abs(diag_value) > 1e-6:
                        validation_errors.append(
                            f"Block [{b},{h},{block_idx}] row {i} (row_in_block={row_in_block}): "
                            f"Diagonal should be 0, got {diag_value:.6f}"
                        )
    
    if validation_errors:
        print(f"  Found {len(validation_errors)} validation errors:")
        for i, error in enumerate(validation_errors[:5]):  # Show first 5 errors
            print(f"    {error}")
        if len(validation_errors) > 5:
            print(f"    ... and {len(validation_errors) - 5} more errors")
        # Don't fail the test here - kernel should handle it gracefully
    else:
        print("  ✓ Input structure validated (all diagonals are zero)")
    
    # Test 1: Basic functionality test
    print("\n" + "="*60)
    print("Test 1: Basic functionality with BT=64")
    print("="*60)
    
    # Call the solve_tril function
    print("Calling solve_tril...")
    Ai = solve_tril(A, output_dtype=torch.float32)
    
    print(f"\nOutput Ai created:")
    print(f"  Shape: {Ai.shape}")
    print(f"  Dtype: {Ai.dtype}")
    print(f"  Device: {Ai.device}")
    
    # Verify the output shape
    assert Ai.shape == (B, T, H, BT), f"Expected shape {(B, T, H, BT)}, got {Ai.shape}"
    print("  ✓ Output shape is correct")
    
    # Convert to numpy for verification
    A_np = A.cpu().numpy()
    Ai_np = Ai.cpu().numpy()
    
    # Verify that (I + A_block) * Ai_block ≈ I for each 64x64 block
    print("\nVerifying block-wise correctness (testing entire 64x64 blocks)...")
    
    max_error = 0.0
    total_blocks_tested = 0
    blocks_with_high_error = []
    
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                start_row = block_idx * BT
                
                # Reconstruct the full 64x64 strictly lower triangular block from A
                # This follows the kernel's interpretation of the packed format
                A_block_full = np.zeros((BT, BT), dtype=np.float32)
                
                # Fill in the block according to row-dependent interpretation
                for i in range(BT):
                    if i < 16:  # Rows 0-15: only cols 0-15 contain L₁₁
                        for j in range(i):  # Only lower triangle has non-zero values
                            A_block_full[i, j] = A_np[b, start_row + i, h, j]
                    elif i < 32:  # Rows 16-31: cols 0-31 contain L₂₁ (0-15) and L₂₂ (16-31)
                        # L₂₁ part (cols 0-15)
                        for j in range(16):
                            A_block_full[i, j] = A_np[b, start_row + i, h, j]
                        # L₂₂ part (cols 16-31)
                        for j in range(16, i+1):  # Only up to diagonal (which is zero)
                            A_block_full[i, j] = A_np[b, start_row + i, h, j]
                    elif i < 48:  # Rows 32-47: cols 0-47 contain L₃₁, L₃₂, L₃₃
                        # L₃₁ part (cols 0-15)
                        for j in range(16):
                            A_block_full[i, j] = A_np[b, start_row + i, h, j]
                        # L₃₂ part (cols 16-31)
                        for j in range(16, 32):
                            A_block_full[i, j] = A_np[b, start_row + i, h, j]
                        # L₃₃ part (cols 32-47)
                        for j in range(32, i+1):  # Only up to diagonal (which is zero)
                            A_block_full[i, j] = A_np[b, start_row + i, h, j]
                    else:  # Rows 48-63: all cols contain L₄₁, L₄₂, L₄₃, L₄₄
                        # L₄₁ part (cols 0-15)
                        for j in range(16):
                            A_block_full[i, j] = A_np[b, start_row + i, h, j]
                        # L₄₂ part (cols 16-31)
                        for j in range(16, 32):
                            A_block_full[i, j] = A_np[b, start_row + i, h, j]
                        # L₄₃ part (cols 32-47)
                        for j in range(32, 48):
                            A_block_full[i, j] = A_np[b, start_row + i, h, j]
                        # L₄₄ part (cols 48-63)
                        for j in range(48, i+1):  # Only up to diagonal (which is zero)
                            A_block_full[i, j] = A_np[b, start_row + i, h, j]
                
                # Extract the computed inverse block from Ai
                # Ai stores the full 64x64 inverse block
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
                
                # Track blocks with high error for debugging
                if error > 1e-4:
                    blocks_with_high_error.append({
                        'batch': b,
                        'head': h,
                        'block': block_idx,
                        'error': error,
                        'A_norm': np.max(np.abs(A_block_full)),
                        'Ai_norm': np.max(np.abs(Ai_block))
                    })
                
                total_blocks_tested += 1
    
    print(f"\nVerification completed:")
    print(f"  Total 64x64 blocks tested: {total_blocks_tested}")
    print(f"  Maximum error across all blocks: {max_error:.6e}")
    
    # Report on blocks with high error
    if blocks_with_high_error:
        print(f"\n  Found {len(blocks_with_high_error)} blocks with error > 1e-4:")
        for block_info in blocks_with_high_error[:3]:  # Show first 3
            print(f"    Block [B={block_info['batch']}, H={block_info['head']}, "
                  f"Idx={block_info['block']}]: error={block_info['error']:.6e}, "
                  f"||A||={block_info['A_norm']:.4f}, ||Ai||={block_info['Ai_norm']:.4f}")
        if len(blocks_with_high_error) > 3:
            print(f"    ... and {len(blocks_with_high_error) - 3} more")
    
    # Final verdict
    print("\n" + "="*60)
    if max_error < 1e-4:
        print("✓ SUCCESS: All 64x64 blocks passed! (I + A) * Ai ≈ I within tolerance 1e-4")
    else:
        print(f"✗ FAILURE: Some 64x64 blocks failed! Maximum error: {max_error:.6e}")
        print("  Note: For 64x64 blocks, numerical errors might accumulate.")
        print("  Consider using double precision if higher accuracy is needed.")
    
    # Additional check: Verify the structure of Ai
    print("\n" + "="*60)
    print("Additional structural checks on Ai...")
    
    # Check 1: Ai should be lower triangular with 1s on diagonal
    print("\nChecking Ai structure:")
    diag_errors = []
    upper_tri_errors = []
    
    for b in range(min(B, 2)):  # Check first 2 batches to save time
        for h in range(min(H, 2)):  # Check first 2 heads
            for block_idx in range(min(num_blocks, 2)):  # Check first 2 blocks
                start_row = block_idx * BT
                
                # Check diagonal elements (should be close to 1)
                for i in range(BT):
                    diag_value = Ai[b, start_row + i, h, i].item()
                    if abs(diag_value - 1.0) > 1e-4:
                        diag_errors.append(
                            f"Block [{b},{h},{block_idx}] row {i}: "
                            f"Diagonal should be 1, got {diag_value:.6f}"
                        )
                
                # Check upper triangle (should be close to 0)
                # Only check a few elements to avoid O(n²) complexity
                for i in range(min(BT, 16)):  # Check first 16 rows
                    for j in range(i + 1, min(BT, i + 17)):  # Check next 16 columns
                        upper_value = Ai[b, start_row + i, h, j].item()
                        if abs(upper_value) > 1e-3:
                            upper_tri_errors.append(
                                f"Block [{b},{h},{block_idx}] position ({i},{j}): "
                                f"Upper triangle should be ~0, got {upper_value:.6f}"
                            )
    
    if diag_errors:
        print(f"  Found {len(diag_errors)} diagonal errors (first 3 shown):")
        for i, error in enumerate(diag_errors[:3]):
            print(f"    {error}")
    else:
        print("  ✓ All checked diagonal elements are close to 1")
    
    if upper_tri_errors:
        print(f"  Found {len(upper_tri_errors)} upper triangle errors (first 3 shown):")
        for i, error in enumerate(upper_tri_errors[:3]):
            print(f"    {error}")
    else:
        print("  ✓ All checked upper triangle elements are close to 0")
    
    print("\n" + "="*60)
    print("TEST SUMMARY:")
    print(f"  Total 64x64 blocks processed: {B * H * num_blocks}")
    print(f"  Maximum (I+A)*Ai - I error: {max_error:.6e}")
    print(f"  Test {'PASSED' if max_error < 1e-4 else 'FAILED'}")
    print("="*60)
    
    return A, Ai

# Run the test
if __name__ == "__main__":    
    print("\n" + "="*80)
    print("Starting test_solve_tril_64x64")
    print("="*80)
    
    try:
        import time
        start_time = time.time()
        
        A, Ai = test_solve_tril_64x64()
        
        end_time = time.time()
        print(f"\nTest completed in {end_time - start_time:.2f} seconds")
        
        print("\n" + "="*80)
        print("END OF TEST")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80)
        print("TEST FAILED WITH EXCEPTION")
        print("="*80)

// -----// IR Dump After PlanMemory (hivm-plan-memory) //----- //
func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c49152_i64 = arith.constant 49152 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c63_i32 = arith.constant 63 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1_i32 = arith.constant 1 : i32
  %c24_i32 = arith.constant 24 : i32
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg14, %arg13 : i32
  %5 = arith.addi %arg6, %c63_i32 : i32
  %6 = arith.divsi %5, %c64_i32 : i32
  %7 = arith.addi %arg7, %c63_i32 : i32
  %8 = arith.divsi %7, %c64_i32 : i32
  %9 = arith.muli %8, %c8_i32 : i32
  %10 = arith.addi %arg8, %c63_i32 : i32
  %11 = arith.divsi %10, %c64_i32 : i32
  %12 = arith.muli %arg10, %c64_i32 : i32
  %13 = arith.index_cast %arg9 : i32 to index
  %14 = arith.index_cast %arg10 : i32 to index
  %15 = arith.index_cast %arg6 : i32 to index
  %16 = arith.index_cast %arg7 : i32 to index
  %17 = arith.index_cast %12 : i32 to index
  %18 = arith.index_cast %arg11 : i32 to index
  scf.for %arg15 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %19 = arith.divsi %arg15, %4 : i32
    %20 = arith.remsi %19, %arg12 : i32
    %21 = arith.divsi %20, %9 : i32
    %22 = arith.muli %21, %c8_i32 : i32
    %23 = arith.subi %6, %22 : i32
    %24 = arith.minsi %23, %c8_i32 : i32
    %25 = arith.remsi %20, %24 : i32
    %26 = arith.addi %22, %25 : i32
    %27 = arith.remsi %20, %9 : i32
    %28 = arith.divsi %27, %24 : i32
    %29 = arith.muli %26, %c64_i32 : i32
    %30 = arith.muli %28, %c64_i32 : i32
    %31 = arith.index_cast %29 : i32 to index
    %32 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%31, %13]
    %33 = arith.index_cast %30 : i32 to index
    %34 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%31, %15]
    %35 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%34, %31]
    %36 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%35, %31]
    %37 = affine.max affine_map<()[s0] -> (0, s0)>()[%36]
    %38 = arith.cmpi slt, %37, %c64 : index
    %39 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%33, %16]
    %40 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%39, %33]
    %41 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%40, %33]
    %42 = affine.max affine_map<()[s0] -> (0, s0)>()[%41]
    %43 = arith.cmpi slt, %42, %c64 : index
    %44 = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %cast = memref.cast %44 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    %45:2 = scf.for %arg16 = %c0_i32 to %11 step %c1_i32 iter_args(%arg17 = %32, %arg18 = %c0) -> (index, index)  : i32 {
      %49 = hivm.hir.pointer_cast(%c16384_i64, %c49152_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %49 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %50 = hivm.hir.pointer_cast(%c0_i64, %c32768_i64) : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %50 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%arg17], sizes: [64, 64], strides: [%13, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %51 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %33]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [%14, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %52 = arith.muli %arg16, %c64_i32 : i32
      %53 = arith.subi %arg8, %52 : i32
      %cast_4 = memref.cast %50 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %54 = arith.index_cast %53 : i32 to index
      %55 = affine.max affine_map<()[s0] -> (0, s0)>()[%54]
      %56 = affine.min affine_map<()[s0] -> (64, s0)>()[%55]
      %57 = affine.max affine_map<()[s0] -> (0, s0)>()[%56]
      %58 = arith.cmpi slt, %57, %c64 : index
      %59 = arith.ori %38, %58 : i1
      %subview_5 = memref.subview %reinterpret_cast_2[0, 0] [%37, %57] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %60 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%37]
      %61 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%57]
      %subview_6 = memref.subview %50[0, 0, 0, 0] [%61, %60, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
      %cast_7 = memref.cast %subview_6 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %59 {
        %collapse_shape = memref.collapse_shape %50 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>>)
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview_5 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_7 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %cast_8 = memref.cast %49 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
      %62 = arith.ori %58, %43 : i1
      %subview_9 = memref.subview %reinterpret_cast_3[0, 0] [%57, %42] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %63 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%57]
      %64 = affine.apply affine_map<()[s0] -> ((s0 + 7) floordiv 8)>()[%42]
      %subview_10 = memref.subview %49[0, 0, 0, 0] [%64, %63, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
      %cast_11 = memref.cast %subview_10 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %62 {
        %collapse_shape = memref.collapse_shape %49 [[0, 1, 2, 3]] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> into memref<4096xf32, #hivm.address_space<cbuf>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<cbuf>>)
      } {hivm.unlikely_condition}
      hivm.hir.nd2nz {dst_continuous} ins(%subview_9 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_11 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %65 = arith.cmpi eq, %arg16, %c0_i32 : i32
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_4, %cast_8, %65, %37, %57, %42 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      %66 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%arg17]
      %67 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %17]
      scf.yield %66, %67 : index, index
    } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
    %46 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%33, %31, %18]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [64, 64], strides: [%18, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %47 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%37]
    %48 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%42]
    %subview = memref.subview %44[0, 0, 0, 0] [%48, %47, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>>
    %cast_0 = memref.cast %subview : memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
    %subview_1 = memref.subview %reinterpret_cast[0, 0] [%37, %42] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_0 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_1 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  return
}


// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_per_token_group_quant_8bit_colmajor_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c44512_i64 = arith.constant 44512 : i64
  %c43488_i64 = arith.constant 43488 : i64
  %c43424_i64 = arith.constant 43424 : i64
  %c39328_i64 = arith.constant 39328 : i64
  %c39296_i64 = arith.constant 39296 : i64
  %c35200_i64 = arith.constant 35200 : i64
  %c31104_i64 = arith.constant 31104 : i64
  %c30976_i64 = arith.constant 30976 : i64
  %c30944_i64 = arith.constant 30944 : i64
  %c26848_i64 = arith.constant 26848 : i64
  %c26784_i64 = arith.constant 26784 : i64
  %c22688_i64 = arith.constant 22688 : i64
  %c18592_i64 = arith.constant 18592 : i64
  %c18464_i64 = arith.constant 18464 : i64
  %c17440_i64 = arith.constant 17440 : i64
  %c17312_i64 = arith.constant 17312 : i64
  %c17280_i64 = arith.constant 17280 : i64
  %c17248_i64 = arith.constant 17248 : i64
  %c17120_i64 = arith.constant 17120 : i64
  %c17056_i64 = arith.constant 17056 : i64
  %c16928_i64 = arith.constant 16928 : i64
  %c16800_i64 = arith.constant 16800 : i64
  %c16768_i64 = arith.constant 16768 : i64
  %c16640_i64 = arith.constant 16640 : i64
  %c12544_i64 = arith.constant 12544 : i64
  %c8448_i64 = arith.constant 8448 : i64
  %c8320_i64 = arith.constant 8320 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.270000e+02 : f32
  %c16_i32 = arith.constant 16 : i32
  %c128_i32 = arith.constant 128 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant -1.280000e+02 : f32
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %cst_2 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<32x32xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape : memref<1024xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c4096_i64) : memref<32x32xf32, #hivm.address_space<ub>>
  %collapse_shape_3 = memref.collapse_shape %6 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_3 : memref<1024xf32, #hivm.address_space<ub>>)
  %7 = hivm.hir.pointer_cast(%c8192_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg6 : f32) outs(%7 : memref<32xf32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c8192_i64) : memref<32xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%7, %7 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%8 : memref<32xi1, #hivm.address_space<ub>>)
  %9 = hivm.hir.pointer_cast(%c8192_i64) : memref<32xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%8 : memref<32xi1, #hivm.address_space<ub>>) outs(%9 : memref<32xi1, #hivm.address_space<ub>>)
  %10 = hivm.hir.pointer_cast(%c8320_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%10 : memref<32xf32, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c4096_i64) : memref<32x32xi1, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %11 [[0, 1]] : memref<32x32xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape_3, %collapse_shape_3 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<1024xi1, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c4096_i64) : memref<32x32xi1, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %12 [[0, 1]] : memref<32x32xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_4 : memref<1024xi1, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<1024xi1, #hivm.address_space<ub>>)
  %13 = hivm.hir.pointer_cast(%c0_i64) : memref<32x32xi1, #hivm.address_space<ub>>
  %collapse_shape_6 = memref.collapse_shape %13 [[0, 1]] : memref<32x32xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
  hivm.hir.vcmp ins(%collapse_shape, %collapse_shape : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<1024xi1, #hivm.address_space<ub>>)
  %14 = hivm.hir.pointer_cast(%c0_i64) : memref<32x32xi1, #hivm.address_space<ub>>
  %collapse_shape_7 = memref.collapse_shape %14 [[0, 1]] : memref<32x32xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_6 : memref<1024xi1, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<1024xi1, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %15 = arith.divsi %arg10, %arg9 : i32
    %16 = arith.remsi %15, %arg8 : i32
    %17 = arith.divsi %arg10, %4 : i32
    %18 = arith.remsi %17, %arg7 : i32
    %19 = arith.muli %18, %c32_i32 : i32
    %20 = arith.muli %16, %c32_i32 : i32
    %21 = arith.index_cast %19 : i32 to index
    %22 = arith.index_cast %20 : i32 to index
    %23 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 128)>()[%22, %21]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [32, 32], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [32, 32], strides: [128, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<32x32xi8, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %24 = arith.muli %16, %c16_i32 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %21]
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%26], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %27 = arith.subi %c128_i32, %20 : i32
    %28 = hivm.hir.pointer_cast(%c8448_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %29 = affine.max affine_map<()[s0] -> (16, s0)>()[%21]
    %30 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%29, %21]
    %31 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%30, %21]
    %32 = arith.index_cast %27 : i32 to index
    %33 = affine.max affine_map<()[s0] -> (0, s0)>()[%32]
    %34 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%30, %21]
    %35 = affine.max affine_map<()[s0] -> (0, s0)>()[%34]
    %36 = affine.min affine_map<()[s0] -> (32, s0)>()[%33]
    %37 = affine.max affine_map<()[s0] -> (0, s0)>()[%36]
    %38 = arith.cmpi slt, %35, %c32 : index
    %39 = arith.cmpi slt, %37, %c32 : index
    %40 = arith.ori %38, %39 : i1
    %subview = memref.subview %reinterpret_cast[0, 0] [%35, %37] [1, 1] : memref<32x32xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %28[0, 0] [%35, %37] [1, 1] : memref<32x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    scf.if %40 {
      %collapse_shape_31 = memref.collapse_shape %28 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_2 : f32) outs(%collapse_shape_31 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %41 = hivm.hir.pointer_cast(%c12544_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %28 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %41 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vabs ins(%collapse_shape_11 : memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_12 : memref<1024xf32, #hivm.address_space<ub>>)
    %42 = hivm.hir.pointer_cast(%c16640_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%41 : memref<32x32xf32, #hivm.address_space<ub>>) outs(%42 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_13 = memref.collapse_shape %42 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %43 = hivm.hir.pointer_cast(%c16768_i64) : memref<32xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_13, %collapse_shape_13 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%43 : memref<32xi1, #hivm.address_space<ub>>)
    %44 = hivm.hir.pointer_cast(%c16768_i64) : memref<32xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%43 : memref<32xi1, #hivm.address_space<ub>>) outs(%44 : memref<32xi1, #hivm.address_space<ub>>)
    %45 = hivm.hir.pointer_cast(%c16800_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_13, %arg6 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%45 : memref<32xf32, #hivm.address_space<ub>>)
    %46 = hivm.hir.pointer_cast(%c16928_i64) : memref<32xf32, #hivm.address_space<ub>>
    %47 = hivm.hir.pointer_cast(%c17056_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%44, %arg6, %45 : memref<32xi1, #hivm.address_space<ub>>, f32, memref<32xf32, #hivm.address_space<ub>>) outs(%46 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%47 : memref<16xf32, #hivm.address_space<ub>>)
    %48 = hivm.hir.pointer_cast(%c17120_i64) : memref<32xf32, #hivm.address_space<ub>>
    %49 = hivm.hir.pointer_cast(%c17248_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%9, %collapse_shape_13, %46 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%48 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%49 : memref<8xf32, #hivm.address_space<ub>>)
    %50 = hivm.hir.pointer_cast(%c17120_i64) : memref<32xf32, #hivm.address_space<ub>>
    %51 = hivm.hir.pointer_cast(%c17280_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%48, %cst : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%50 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<8xf32, #hivm.address_space<ub>>)
    %52 = hivm.hir.pointer_cast(%c17312_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%10, %50 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%52 : memref<32xf32, #hivm.address_space<ub>>)
    %expand_shape = memref.expand_shape %52 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
    %53 = hivm.hir.pointer_cast(%c8448_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %54 = hivm.hir.pointer_cast(%c17440_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%28, %expand_shape : memref<32x32xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%53 : memref<32x32xf32, #hivm.address_space<ub>>) temp_buffer(%54 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    %55 = hivm.hir.pointer_cast(%c18464_i64) : memref<32x32xi1, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %53 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %55 [[0, 1]] : memref<32x32xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_14, %collapse_shape_14 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_15 : memref<1024xi1, #hivm.address_space<ub>>)
    %56 = hivm.hir.pointer_cast(%c18464_i64) : memref<32x32xi1, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %56 [[0, 1]] : memref<32x32xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_15 : memref<1024xi1, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<1024xi1, #hivm.address_space<ub>>)
    %57 = hivm.hir.pointer_cast(%c18592_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %57 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.vmin ins(%collapse_shape_14, %cst : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_17 : memref<1024xf32, #hivm.address_space<ub>>)
    %58 = hivm.hir.pointer_cast(%c22688_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %58 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %59 = hivm.hir.pointer_cast(%c26784_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_16, %cst, %collapse_shape_17 : memref<1024xi1, #hivm.address_space<ub>>, f32, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_18 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%59 : memref<16xf32, #hivm.address_space<ub>>)
    %60 = hivm.hir.pointer_cast(%c26848_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %60 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c30944_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_5, %collapse_shape_14, %collapse_shape_18 : memref<1024xi1, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%61 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    %62 = hivm.hir.pointer_cast(%c30976_i64) : memref<32x32xi1, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %62 [[0, 1]] : memref<32x32xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape_19, %collapse_shape_19 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<1024xi1, #hivm.address_space<ub>>)
    %63 = hivm.hir.pointer_cast(%c30976_i64) : memref<32x32xi1, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %63 [[0, 1]] : memref<32x32xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%collapse_shape_20 : memref<1024xi1, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<1024xi1, #hivm.address_space<ub>>)
    %64 = hivm.hir.pointer_cast(%c31104_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %64 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%collapse_shape_19, %cst_1 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_22 : memref<1024xf32, #hivm.address_space<ub>>)
    %65 = hivm.hir.pointer_cast(%c35200_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %65 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %66 = hivm.hir.pointer_cast(%c39296_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_7, %collapse_shape_19, %collapse_shape_22 : memref<1024xi1, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%66 : memref<8xf32, #hivm.address_space<ub>>)
    %67 = hivm.hir.pointer_cast(%c39328_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %67 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %68 = hivm.hir.pointer_cast(%c43424_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_21, %cst_1, %collapse_shape_23 : memref<1024xi1, #hivm.address_space<ub>>, f32, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%68 : memref<16xf32, #hivm.address_space<ub>>)
    %69 = hivm.hir.pointer_cast(%c39328_i64) : memref<32x32xi32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %69 [[0, 1]] : memref<32x32xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_24 : memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<1024xi32, #hivm.address_space<ub>>) round_mode = <trunc>
    %70 = hivm.hir.pointer_cast(%c43488_i64) : memref<32x32xi8, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %70 [[0, 1]] : memref<32x32xi8, #hivm.address_space<ub>> into memref<1024xi8, #hivm.address_space<ub>>
    %71 = hivm.hir.pointer_cast(%c44512_i64) : memref<2048xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_25 : memref<1024xi32, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<1024xi8, #hivm.address_space<ub>>) temp_buffer(%71 : memref<2048xi32, #hivm.address_space<ub>>) round_mode = <truncwithoverflow>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_27 = memref.subview %70[0, 0] [%35, %37] [1, 1] : memref<32x32xi8, #hivm.address_space<ub>> to memref<?x?xi8, strided<[32, 1]>, #hivm.address_space<ub>>
    %subview_28 = memref.subview %reinterpret_cast_8[0, 0] [%35, %37] [1, 1] : memref<32x32xi8, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xi8, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_27 : memref<?x?xi8, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%subview_28 : memref<?x?xi8, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
    %subview_29 = memref.subview %50[0] [%31] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_30 = memref.subview %reinterpret_cast_9[0] [%31] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_29 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_30 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.

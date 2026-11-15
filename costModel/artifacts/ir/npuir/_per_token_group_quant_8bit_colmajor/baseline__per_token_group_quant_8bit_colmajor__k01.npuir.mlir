// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_per_token_group_quant_8bit_colmajor(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: f32, %arg13: f32, %arg14: f32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<18xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c16512_i64 = arith.constant 16512 : i64
  %c15200_i64 = arith.constant 15200 : i64
  %c7008_i64 = arith.constant 7008 : i64
  %c15488_i64 = arith.constant 15488 : i64
  %c5984_i64 = arith.constant 5984 : i64
  %c1888_i64 = arith.constant 1888 : i64
  %c1824_i64 = arith.constant 1824 : i64
  %c1696_i64 = arith.constant 1696 : i64
  %c1664_i64 = arith.constant 1664 : i64
  %c1536_i64 = arith.constant 1536 : i64
  %c1408_i64 = arith.constant 1408 : i64
  %c1376_i64 = arith.constant 1376 : i64
  %c1344_i64 = arith.constant 1344 : i64
  %c1216_i64 = arith.constant 1216 : i64
  %c1152_i64 = arith.constant 1152 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c896_i64 = arith.constant 896 : i64
  %c864_i64 = arith.constant 864 : i64
  %c736_i64 = arith.constant 736 : i64
  %c608_i64 = arith.constant 608 : i64
  %c576_i64 = arith.constant 576 : i64
  %c15360_i64 = arith.constant 15360 : i64
  %c448_i64 = arith.constant 448 : i64
  %c416_i64 = arith.constant 416 : i64
  %c384_i64 = arith.constant 384 : i64
  %c352_i64 = arith.constant 352 : i64
  %c320_i64 = arith.constant 320 : i64
  %c288_i64 = arith.constant 288 : i64
  %c256_i64 = arith.constant 256 : i64
  %c224_i64 = arith.constant 224 : i64
  %c192_i64 = arith.constant 192 : i64
  %c160_i64 = arith.constant 160 : i64
  %c128_i64 = arith.constant 128 : i64
  %c15232_i64 = arith.constant 15232 : i64
  %c0_i64 = arith.constant 0 : i64
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg15, %arg16 : i32
  %1 = arith.muli %0, %arg17 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg17, %arg16 : i32
  %5 = arith.addi %arg8, %arg6 : i32
  %6 = arith.subi %5, %c1_i32 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg18 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %7 = arith.divsi %arg18, %4 : i32
    %8 = arith.remsi %7, %arg15 : i32
    %9 = arith.divsi %6, %arg6 : i32
    %10 = arith.muli %8, %c16_i32 : i32
    %11 = arith.addi %10, %c16_i32 : i32
    %12 = arith.minsi %11, %arg7 : i32
    scf.for %arg19 = %10 to %12 step %c1_i32  : i32 {
      scf.for %arg20 = %c0_i32 to %9 step %c1_i32  : i32 {
        %13 = arith.index_cast %arg20 : i32 to index
        %14 = arith.index_cast %c0_i32 : i32 to index
        %15 = arith.index_cast %9 : i32 to index
        %16 = arith.index_cast %c1_i32 : i32 to index
        %17 = arith.index_cast %arg19 : i32 to index
        %18 = arith.index_cast %10 : i32 to index
        %19 = arith.index_cast %12 : i32 to index
        %20 = arith.index_cast %c1_i32 : i32 to index
        %21 = arith.index_cast %arg18 : i32 to index
        %22 = arith.index_cast %3 : i32 to index
        %23 = arith.index_cast %1 : i32 to index
        %24 = arith.index_cast %c48_i32 : i32 to index
        %25 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s7) * ((-s1 + s2 + s3 - 1) floordiv s3) + ((s8 - s9) floordiv s10) * (((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7))) mod 2)>()[%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %24]
        %26 = arith.index_cast %25 : index to i1
        %c0_i64_0 = arith.constant 0 : i64
        %c1_i64 = arith.constant 1 : i64
        %27 = arith.select %26, %c0_i64_0, %c1_i64 : i64
        %28 = hivm.hir.pointer_cast(%c15200_i64, %c16512_i64) : memref<1xf32, #hivm.address_space<ub>>
        annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
        %29 = arith.muli %arg20, %arg6 : i32
        %30 = arith.addi %29, %arg6 : i32
        %31 = arith.minsi %30, %arg8 : i32
        %32 = arith.subi %31, %29 : i32
        %33 = arith.cmpi sgt, %32, %c0_i32 : i32
        scf.if %33 {
          %34 = arith.muli %arg19, %arg9 : i32
          %35 = arith.muli %arg19, %arg10 : i32
          %36 = arith.muli %arg20, %arg11 : i32
          %37 = scf.for %arg21 = %c0_i32 to %32 step %c32_i32 iter_args(%arg22 = %arg12) -> (f32)  : i32 {
            %42 = hivm.hir.pointer_cast(%c0_i64, %c15232_i64) : memref<32xf32, #hivm.address_space<ub>>
            annotation.mark %42 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
            %43 = arith.addi %arg21, %c32_i32 : i32
            %44 = arith.minsi %43, %32 : i32
            %45 = arith.subi %44, %arg21 : i32
            %46 = arith.cmpi sgt, %45, %c0_i32 : i32
            %47 = scf.if %46 -> (f32) {
              %48 = arith.index_cast %34 : i32 to index
              %49 = arith.index_cast %29 : i32 to index
              %50 = arith.index_cast %arg21 : i32 to index
              %51 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%50, %48, %49]
              %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%51], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
              %52 = arith.index_cast %45 : i32 to index
              %53 = affine.max affine_map<()[s0] -> (0, s0)>()[%52]
              %54 = affine.min affine_map<()[s0] -> (32, s0)>()[%53]
              %55 = arith.cmpi slt, %54, %c32 : index
              %subview = memref.subview %reinterpret_cast_1[0] [%54] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
              %subview_2 = memref.subview %42[0] [%54] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
              scf.if %55 {
                hivm.hir.vbrc ins(%cst : f32) outs(%42 : memref<32xf32, #hivm.address_space<ub>>)
                hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
                hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
              } {hivm.unlikely_condition}
              hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
              hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
              hivm.hir.vabs ins(%42 : memref<32xf32, #hivm.address_space<ub>>) outs(%42 : memref<32xf32, #hivm.address_space<ub>>)
              %56 = hivm.hir.pointer_cast(%c128_i64) : memref<1xf32, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vreduce <max> ins(%42 : memref<32xf32, #hivm.address_space<ub>>) outs(%56 : memref<1xf32, #hivm.address_space<ub>>) reduce_dims = [0]
              hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
              %57 = memref.load %56[%c0] : memref<1xf32, #hivm.address_space<ub>>
              %58 = hivm.hir.pointer_cast(%c160_i64) : memref<1xf32, #hivm.address_space<ub>>
              memref.store %arg22, %58[%c0] : memref<1xf32, #hivm.address_space<ub>>
              hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
              %59 = hivm.hir.pointer_cast(%c192_i64) : memref<1xf32, #hivm.address_space<ub>>
              memref.store %57, %59[%c0] : memref<1xf32, #hivm.address_space<ub>>
              hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
              %60 = hivm.hir.pointer_cast(%c224_i64) : memref<1xi1, #hivm.address_space<ub>>
              hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
              hivm.hir.vcmp ins(%58, %58 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%60 : memref<1xi1, #hivm.address_space<ub>>)
              %61 = hivm.hir.pointer_cast(%c224_i64) : memref<1xi1, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vnot ins(%60 : memref<1xi1, #hivm.address_space<ub>>) outs(%61 : memref<1xi1, #hivm.address_space<ub>>)
              %62 = hivm.hir.pointer_cast(%c256_i64) : memref<1xi1, #hivm.address_space<ub>>
              hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
              hivm.hir.vcmp ins(%59, %59 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%62 : memref<1xi1, #hivm.address_space<ub>>)
              %63 = hivm.hir.pointer_cast(%c256_i64) : memref<1xi1, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vnot ins(%62 : memref<1xi1, #hivm.address_space<ub>>) outs(%63 : memref<1xi1, #hivm.address_space<ub>>)
              %64 = hivm.hir.pointer_cast(%c288_i64) : memref<1xf32, #hivm.address_space<ub>>
              %65 = arith.maximumf %arg22, %57 : f32
              memref.store %65, %64[%c0] : memref<1xf32, #hivm.address_space<ub>>
              hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
              %66 = hivm.hir.pointer_cast(%c320_i64) : memref<1xf32, #hivm.address_space<ub>>
              %67 = hivm.hir.pointer_cast(%c352_i64) : memref<8xf32, #hivm.address_space<ub>>
              hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
              hivm.hir.vsel ins(%61, %59, %64 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%66 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%67 : memref<8xf32, #hivm.address_space<ub>>)
              %68 = hivm.hir.pointer_cast(%c384_i64) : memref<1xf32, #hivm.address_space<ub>>
              %69 = hivm.hir.pointer_cast(%c416_i64) : memref<8xf32, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vsel ins(%63, %58, %66 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%68 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%69 : memref<8xf32, #hivm.address_space<ub>>)
              hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
              %70 = memref.load %68[%c0] : memref<1xf32, #hivm.address_space<ub>>
              scf.yield %70 : f32
            } else {
              scf.yield %arg22 : f32
            } {pipeline.veconly}
            scf.yield %47 : f32
          }
          %38 = arith.divf %37, %arg14 : f32
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          scf.for %arg21 = %c0_i32 to %32 step %c32_i32  : i32 {
            %42 = arith.index_cast %arg21 : i32 to index
            %43 = arith.index_cast %c0_i32 : i32 to index
            %44 = arith.index_cast %32 : i32 to index
            %45 = arith.index_cast %c32_i32 : i32 to index
            %46 = arith.index_cast %arg20 : i32 to index
            %47 = arith.index_cast %c0_i32 : i32 to index
            %48 = arith.index_cast %9 : i32 to index
            %49 = arith.index_cast %c1_i32 : i32 to index
            %50 = arith.index_cast %arg19 : i32 to index
            %51 = arith.index_cast %10 : i32 to index
            %52 = arith.index_cast %12 : i32 to index
            %53 = arith.index_cast %c1_i32 : i32 to index
            %54 = arith.index_cast %arg18 : i32 to index
            %55 = arith.index_cast %3 : i32 to index
            %56 = arith.index_cast %1 : i32 to index
            %57 = arith.index_cast %c48_i32 : i32 to index
            %58 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s7) * ((-s1 + s2 + s3 - 1) floordiv s3) + ((s8 - s9) floordiv s11) * (((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7)) + ((s12 - s13) floordiv s14) * ((((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7)) * ((-s9 + s10 + s11 - 1) floordiv s11))) mod 2)>()[%42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %57]
            %59 = arith.index_cast %58 : index to i1
            %c0_i64_1 = arith.constant 0 : i64
            %c1_i64_2 = arith.constant 1 : i64
            %60 = arith.select %59, %c0_i64_1, %c1_i64_2 : i64
            %61 = hivm.hir.pointer_cast(%c5984_i64, %c15488_i64) : memref<1024xi8, #hivm.address_space<ub>>
            annotation.mark %61 {hivm.multi_buffer = 2 : i32} : memref<1024xi8, #hivm.address_space<ub>>
            %62 = hivm.hir.pointer_cast(%c448_i64, %c15360_i64) : memref<32xf32, #hivm.address_space<ub>>
            annotation.mark %62 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
            %63 = arith.addi %arg21, %c32_i32 : i32
            %64 = arith.minsi %63, %32 : i32
            %65 = arith.subi %64, %arg21 : i32
            %66 = arith.cmpi sgt, %65, %c0_i32 : i32
            scf.if %66 {
              %67 = arith.index_cast %34 : i32 to index
              %68 = arith.index_cast %29 : i32 to index
              %69 = arith.index_cast %arg21 : i32 to index
              %70 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%69, %67, %68]
              %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%70], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
              %71 = arith.index_cast %65 : i32 to index
              %72 = affine.max affine_map<()[s0] -> (0, s0)>()[%71]
              %73 = affine.min affine_map<()[s0] -> (32, s0)>()[%72]
              %74 = arith.cmpi slt, %73, %c32 : index
              %subview = memref.subview %reinterpret_cast_3[0] [%73] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
              %subview_4 = memref.subview %62[0] [%73] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
              scf.if %74 {
                hivm.hir.vbrc ins(%cst : f32) outs(%62 : memref<32xf32, #hivm.address_space<ub>>)
                hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
                hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
              } {hivm.unlikely_condition}
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %60]
              hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
              hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
              %75 = hivm.hir.pointer_cast(%c576_i64) : memref<8xf32, #hivm.address_space<ub>>
              hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
              hivm.hir.vdiv ins(%62, %38 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%62 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%75 : memref<8xf32, #hivm.address_space<ub>>)
              %76 = hivm.hir.pointer_cast(%c608_i64) : memref<32xf32, #hivm.address_space<ub>>
              hivm.hir.vbrc ins(%arg13 : f32) outs(%76 : memref<32xf32, #hivm.address_space<ub>>)
              %77 = hivm.hir.pointer_cast(%c736_i64) : memref<32xf32, #hivm.address_space<ub>>
              hivm.hir.vbrc ins(%arg14 : f32) outs(%77 : memref<32xf32, #hivm.address_space<ub>>)
              %78 = hivm.hir.pointer_cast(%c864_i64) : memref<32xi1, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vcmp ins(%62, %62 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%78 : memref<32xi1, #hivm.address_space<ub>>)
              %79 = hivm.hir.pointer_cast(%c864_i64) : memref<32xi1, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vnot ins(%78 : memref<32xi1, #hivm.address_space<ub>>) outs(%79 : memref<32xi1, #hivm.address_space<ub>>)
              %80 = hivm.hir.pointer_cast(%c736_i64) : memref<32xi1, #hivm.address_space<ub>>
              hivm.hir.vcmp ins(%77, %77 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%80 : memref<32xi1, #hivm.address_space<ub>>)
              %81 = hivm.hir.pointer_cast(%c736_i64) : memref<32xi1, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vnot ins(%80 : memref<32xi1, #hivm.address_space<ub>>) outs(%81 : memref<32xi1, #hivm.address_space<ub>>)
              %82 = hivm.hir.pointer_cast(%c896_i64) : memref<32xf32, #hivm.address_space<ub>>
              hivm.hir.vmin ins(%62, %arg14 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%82 : memref<32xf32, #hivm.address_space<ub>>)
              %83 = hivm.hir.pointer_cast(%c1024_i64) : memref<32xf32, #hivm.address_space<ub>>
              %84 = hivm.hir.pointer_cast(%c1152_i64) : memref<16xf32, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vsel ins(%79, %arg14, %82 : memref<32xi1, #hivm.address_space<ub>>, f32, memref<32xf32, #hivm.address_space<ub>>) outs(%83 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%84 : memref<16xf32, #hivm.address_space<ub>>)
              %85 = hivm.hir.pointer_cast(%c1216_i64) : memref<32xf32, #hivm.address_space<ub>>
              %86 = hivm.hir.pointer_cast(%c1344_i64) : memref<8xf32, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vsel ins(%81, %62, %83 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%85 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%86 : memref<8xf32, #hivm.address_space<ub>>)
              hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %60]
              %87 = hivm.hir.pointer_cast(%c608_i64) : memref<32xi1, #hivm.address_space<ub>>
              hivm.hir.vcmp ins(%76, %76 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%87 : memref<32xi1, #hivm.address_space<ub>>)
              %88 = hivm.hir.pointer_cast(%c608_i64) : memref<32xi1, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vnot ins(%87 : memref<32xi1, #hivm.address_space<ub>>) outs(%88 : memref<32xi1, #hivm.address_space<ub>>)
              %89 = hivm.hir.pointer_cast(%c1376_i64) : memref<32xi1, #hivm.address_space<ub>>
              hivm.hir.vcmp ins(%85, %85 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%89 : memref<32xi1, #hivm.address_space<ub>>)
              %90 = hivm.hir.pointer_cast(%c1376_i64) : memref<32xi1, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vnot ins(%89 : memref<32xi1, #hivm.address_space<ub>>) outs(%90 : memref<32xi1, #hivm.address_space<ub>>)
              %91 = hivm.hir.pointer_cast(%c1408_i64) : memref<32xf32, #hivm.address_space<ub>>
              hivm.hir.vmax ins(%85, %arg13 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%91 : memref<32xf32, #hivm.address_space<ub>>)
              %92 = hivm.hir.pointer_cast(%c1536_i64) : memref<32xf32, #hivm.address_space<ub>>
              %93 = hivm.hir.pointer_cast(%c1664_i64) : memref<8xf32, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vsel ins(%88, %85, %91 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%92 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%93 : memref<8xf32, #hivm.address_space<ub>>)
              %94 = hivm.hir.pointer_cast(%c1696_i64) : memref<32xf32, #hivm.address_space<ub>>
              %95 = hivm.hir.pointer_cast(%c1824_i64) : memref<16xf32, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vsel ins(%90, %arg13, %92 : memref<32xi1, #hivm.address_space<ub>>, f32, memref<32xf32, #hivm.address_space<ub>>) outs(%94 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%95 : memref<16xf32, #hivm.address_space<ub>>)
              %96 = hivm.hir.pointer_cast(%c1888_i64) : memref<1024xi32, #hivm.address_space<ub>>
              %subview_5 = memref.subview %96[0] [32] [1] : memref<1024xi32, #hivm.address_space<ub>> to memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vcast ins(%94 : memref<32xf32, #hivm.address_space<ub>>) outs(%subview_5 : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>) round_mode = <trunc>
              %subview_6 = memref.subview %61[0] [32] [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<32xi8, strided<[1]>, #hivm.address_space<ub>>
              %97 = hivm.hir.pointer_cast(%c7008_i64) : memref<2048xi32, #hivm.address_space<ub>>
              hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %60]
              hivm.hir.pipe_barrier[<PIPE_V>]
              hivm.hir.vcast ins(%subview_5 : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<32xi8, strided<[1]>, #hivm.address_space<ub>>) temp_buffer(%97 : memref<2048xi32, #hivm.address_space<ub>>) round_mode = <truncwithoverflow>
              hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
              %98 = arith.index_cast %35 : i32 to index
              %99 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%69, %98, %68]
              %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%99], sizes: [32], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<32xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
              %subview_8 = memref.subview %subview_6[0] [%73] [1] : memref<32xi8, strided<[1]>, #hivm.address_space<ub>> to memref<?xi8, strided<[1]>, #hivm.address_space<ub>>
              %subview_9 = memref.subview %reinterpret_cast_7[0] [%73] [1] : memref<32xi8, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
              hivm.hir.pipe_barrier[<PIPE_MTE3>]
              hivm.hir.store ins(%subview_8 : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>)
              hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %60]
            } {pipeline.veconly}
          }
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %27]
          memref.store %38, %28[%c0] : memref<1xf32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          %39 = arith.index_cast %36 : i32 to index
          %40 = arith.index_cast %arg19 : i32 to index
          %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%39, %40]
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%41], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          hivm.hir.store ins(%28 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %27]
        } {pipeline.veconly}
      } {pipeline.veconly}
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.

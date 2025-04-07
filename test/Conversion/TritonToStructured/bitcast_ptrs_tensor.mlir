// RUN: triton-shared-opt --triton-to-structured --canonicalize --cse %s | FileCheck %s

module {
  tt.func @kernel(
    %out_ptr: !tt.ptr<i1>,
    %num_elements: i32,
    %value: i8
  ) -> () {
    // compute offsets.
    %c512 = arith.constant 512 : i32
    %pid = tt.get_program_id x : i32
    %0 = arith.muli %pid, %c512 : i32
    %1 = tt.splat %0 : i32 -> tensor<512xi32>
    %2 = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32>
    %3 = arith.addi %1, %2 : tensor<512xi32>
    // compute mask.
    %4 = tt.splat %num_elements : i32 -> tensor<512xi32>
    %5 = arith.cmpi slt, %3, %4 : tensor<512xi32>
    // ptr arithmetics.
    %6 = tt.splat %out_ptr : !tt.ptr<i1> -> tensor<512x!tt.ptr<i1>>
    %7 = tt.addptr %6, %3 : tensor<512x!tt.ptr<i1>>, tensor<512xi32>
    // reinterpret cast ptr data type.
    %8 = tt.bitcast %7 : tensor<512x!tt.ptr<i1>> -> tensor<512x!tt.ptr<i8>>
    // store.
    %9 = tt.splat %value : i8 -> tensor<512xi8>
    tt.store %8, %9, %5: tensor<512x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:   tt.func @kernel(
// CHECK-SAME:      [[PARAM_0_:%.+]]: !tt.ptr<i1>,
// CHECK-SAME:      [[PARAM_1_:%.+]]: i32,
// CHECK-SAME:      [[PARAM_2_:%.+]]: i8
// CHECK-SAME:    ) {
// CHECK-DAG:     [[CST_512_index_:%.+]] = arith.constant 512 : index
// CHECK-DAG:     [[CST_512_i32_:%.+]] = arith.constant 512 : i32
// CHECK-DAG:     [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:         [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_512_i32_]] : i32
// CHECK:         [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:     [[VAR_3_:%.+]] = tt.bitcast [[PARAM_0_]] : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK:         [[VAR_4_:%.+]] = tts.make_tptr [[VAR_3_]] to sizes: [512], strides: [1], offsets: {{.}}[[VAR_2_]]{{.}}, shape: [0], order: [] : <i8> to tensor<512x!tt.ptr<i8>>
// CHECK:         [[VAR_5_:%.+]] = tt.splat [[PARAM_2_]] : i8 -> tensor<512xi8>
// CHECK:         [[VAR_6_:%.+]] = arith.addi [[VAR_2_]], [[CST_512_index_]] : index
// CHECK:         [[VAR_7_:%.+]] = arith.index_cast [[PARAM_1_]] : i32 to index
// CHECK:         [[VAR_8_:%.+]] = arith.minsi [[VAR_6_]], [[VAR_7_]] : index
// CHECK:         [[VAR_9_:%.+]] = arith.maxsi [[VAR_8_]], [[VAR_2_]] : index
// CHECK:         [[VAR_10_:%.+]] = arith.subi [[VAR_9_]], [[VAR_2_]] : index
// CHECK:         "tts.store"([[VAR_4_]], [[VAR_5_]], [[VAR_10_]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<512x!tt.ptr<i8>>, tensor<512xi8>, index) -> ()
// CHECK:         return
// CHECK:       }

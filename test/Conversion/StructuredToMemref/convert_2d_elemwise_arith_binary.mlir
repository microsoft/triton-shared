// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<f32>,
    %b : !tt.ptr<f32>,
    %c : !tt.ptr<f32>,
    %d : !tt.ptr<f32>
  ) -> () {
        // offset calculations
        %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %moff = tt.broadcast %1 : tensor<128x1xi32> -> tensor<128x128xi32>
        %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
        %koff = tt.broadcast %4 : tensor<1x128xi32> -> tensor<128x128xi32>
        %mkoff = arith.addi %moff, %koff : tensor<128x128xi32>
        // a pointer
        %8 = tt.splat %a : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
        %9 = tt.addptr %8, %mkoff : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
        // b pointer
        %18 = tt.splat %b : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
        %19 = tt.addptr %18, %mkoff : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
        %af = tt.load %9 : tensor<128x128x!tt.ptr<f32>>
        %bf = tt.load %19 : tensor<128x128x!tt.ptr<f32>>
        %res0 = arith.addf %af, %bf : tensor<128x128xf32>
        %res1 = arith.subf %af, %bf : tensor<128x128xf32>
        // c and d pointers, intentionally splat the base pointer for brevity
        %c_out = tt.splat %c : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
        %d_out = tt.splat %d : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
        tt.store %c_out, %res0 : tensor<128x128x!tt.ptr<f32>>
        tt.store %d_out, %res1 : tensor<128x128x!tt.ptr<f32>>
        tt.return
    }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: memref<*xf32>, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<128x128xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<128x128xf32, strided<[1, 1]>> to memref<128x128xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128x128xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<128x128xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_0_]], [[RES_1_]] : memref<128x128xf32, strided<[1, 1]>> to memref<128x128xf32>
// CHECK:           [[VAR_1_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<128x128xf32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_0_]], [[VAR_1_]] : tensor<128x128xf32>, tensor<128x128xf32>) outs([[VAR_0_]] : tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32, [[IN_2_:%.+]]: f32):
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[IN_0_]], [[IN_1_]] : f32
// CHECK:             linalg.yield [[VAR_4_]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           [[VAR_3_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_0_]], [[VAR_1_]] : tensor<128x128xf32>, tensor<128x128xf32>) outs([[VAR_0_]] : tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_3_:%.+]]: f32, [[IN_4_:%.+]]: f32, [[IN_5_:%.+]]: f32):
// CHECK:             [[VAR_4_1_:%.+]] = arith.subf [[IN_3_]], [[IN_4_]] : f32
// CHECK:             linalg.yield [[VAR_4_1_]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           [[VAR_cast_:%.+]] = memref.cast [[PARAM_2_]] : memref<*xf32> to memref<?xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 128 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 128 {
// CHECK:               [[VAR_extracted_:%.+]] = tensor.extract [[VAR_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : tensor<128x128xf32>
// CHECK:               memref.store [[VAR_extracted_]], [[VAR_cast_]]{{.}}[[CST_0_]]{{.}} : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAR_cast_2_:%.+]] = memref.cast [[PARAM_3_]] : memref<*xf32> to memref<?xf32>
// CHECK:           affine.for [[I_2_:%.+]] = 0 to 128 {
// CHECK:             affine.for [[I_3_:%.+]] = 0 to 128 {
// CHECK:               [[VAR_extracted_1_:%.+]] = tensor.extract [[VAR_3_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : tensor<128x128xf32>
// CHECK:               memref.store [[VAR_extracted_1_]], [[VAR_cast_2_]]{{.}}[[CST_0_]]{{.}} : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return

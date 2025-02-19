// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
    tt.func @kernel(%fin : f32,
                    %bin : bf16,
                    %save_ptr0 : !tt.ptr<f32>,
                    %save_ptr1 : !tt.ptr<bf16>) -> () {
        %0 = tt.splat %fin : f32 -> tensor<1024xf32>
        %1 = tt.splat %bin : bf16 -> tensor<128x256xbf16>
        // save pointers, intentionally splat the base pointer for brevity
        %save0 = tt.splat %save_ptr0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %save1 = tt.splat %save_ptr1 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
        tt.store %save0, %0 : tensor<1024x!tt.ptr<f32>>
        tt.store %save1, %1 : tensor<128x256x!tt.ptr<bf16>>
        tt.return
    }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: f32, [[PARAM_1_:%.+]]: bf16, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: memref<*xbf16>, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_cast_:%.+]] = memref.cast [[PARAM_2_]] : memref<*xf32> to memref<?xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 1024 {
// CHECK:             memref.store [[PARAM_0_]], [[VAR_cast_]]{{.}}[[CST_0_]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           [[VAR_cast_0_:%.+]] = memref.cast [[PARAM_3_]] : memref<*xbf16> to memref<?xbf16>
// CHECK:           affine.for [[I_1_:%.+]] = 0 to 128 {
// CHECK:             affine.for [[I_2_:%.+]] = 0 to 256 {
// CHECK:               memref.store [[PARAM_1_]], [[VAR_cast_0_]]{{.}}[[CST_0_]]{{.}} : memref<?xbf16>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

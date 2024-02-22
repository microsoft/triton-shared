// RUN: triton-shared-opt --split-input-file --triton-to-structured --canonicalize --triton-arith-to-linalg --structured-to-memref %s | FileCheck %s
module {
    tt.func @kernel(%fin : f32,
                    %bin : bf16,
                    %save0 : tensor<1024x!tt.ptr<f32>>,
                    %save1 : tensor<128x256x!tt.ptr<bf16>>) -> () {
        %0 = tt.splat %fin : f32 -> tensor<1024xf32>
        %1 = tt.splat %bin : bf16 -> tensor<128x256xbf16>
        tt.store %save0, %0 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
        tt.store %save1, %1 {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xbf16>
        tt.return
    }
}
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: f32, [[PARAM_1_:%.+]]: bf16, [[PARAM_2_:%.+]]: tensor<1024x!tt.ptr<f32, 1>>, [[PARAM_3_:%.+]]: tensor<128x256x!tt.ptr<bf16, 1>>, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[PARAM_0_]] : f32) outs([[VAR_0_]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<128x256xbf16>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_1_]] : bf16) outs([[VAR_2_]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           tt.store [[PARAM_2_]], [[VAR_1_]] {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
// CHECK:           tt.store [[PARAM_3_]], [[VAR_3_]] {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xbf16>
// CHECK:           return
// CHECK:         }

/home/nhat/github/triton_old/third_party/triton_shared/wip/add_linalg.mlir:3:3: error: Dialect `func' not found for custom op 'func.func' 
  func.func @add_kernel_01234(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
  ^
/home/nhat/github/triton_old/third_party/triton_shared/wip/add_linalg.mlir:3:3: note: Registered dialects: arith, builtin, cf, gpu, math, scf, triton_gpu, tt, tts, ttx ; for more info on dialect registration see https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management

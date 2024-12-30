module {
  func.func @addptr(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c2_i32 = arith.constant 2 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    scf.for %arg8 = %c0_i32 to %c10_i32 step %c2_i32  : i32 {
      %0 = arith.extsi %arg8 : i32 to i64
      %1 = arith.addi %0, %c1_i64 : i64
      %2 = arith.addi %1, %c1_i64 : i64
      %3 = arith.extsi %arg8 : i32 to i64
      %4 = arith.addi %3, %c1_i64 : i64
      %5 = arith.addi %4, %c1_i64 : i64
      %6 = "tts.create_ptr"(%arg0, %1) : (!tt.ptr<f32>, i64) -> !tt.ptr<f32>
      %7 = tt.load %6 : !tt.ptr<f32>
      %8 = "tts.create_ptr"(%arg0, %2) : (!tt.ptr<f32>, i64) -> !tt.ptr<f32>
      %9 = tt.load %8 : !tt.ptr<f32>
      %10 = "tts.create_ptr"(%arg1, %4) : (!tt.ptr<f32>, i64) -> !tt.ptr<f32>
      tt.store %10, %7 : !tt.ptr<f32>
      %11 = "tts.create_ptr"(%arg1, %5) : (!tt.ptr<f32>, i64) -> !tt.ptr<f32>
      tt.store %11, %9 : !tt.ptr<f32>
    }
    return
  }
}
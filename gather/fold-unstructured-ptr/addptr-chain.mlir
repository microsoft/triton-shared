module {
  tt.func public @addptr(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %1 = arith.index_cast %c1_i32 : i32 to index
    %2 = arith.index_cast %c1_i32 : i32 to index
    %3 = arith.index_cast %c1_i32 : i32 to index
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c2_i32 = arith.constant 2 : i32
    %4 = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
    %5 = tt.addptr %arg1, %c1_i32 : !tt.ptr<f32>, i32
    scf.for %arg2 = %c0_i32 to %c10_i32 step %c2_i32  : i32 {
      %6 = arith.index_cast %arg2 : i32 to index
      %7 = arith.addi %3, %6 : index
      %8 = tt.addptr %4, %arg2 : !tt.ptr<f32>, i32
      %9 = arith.addi %7, %1 : index
      %10 = tt.addptr %8, %c1_i32 : !tt.ptr<f32>, i32
      %11 = arith.index_cast %arg2 : i32 to index
      %12 = arith.addi %2, %11 : index
      %13 = tt.addptr %5, %arg2 : !tt.ptr<f32>, i32
      %14 = arith.addi %12, %0 : index
      %15 = tt.addptr %13, %c1_i32 : !tt.ptr<f32>, i32
      %16 = tt.load %8 : !tt.ptr<f32>
      %17 = tt.load %10 : !tt.ptr<f32>
      tt.store %13, %16 : !tt.ptr<f32>
      tt.store %15, %17 : !tt.ptr<f32>
    }
    tt.return
  }
}
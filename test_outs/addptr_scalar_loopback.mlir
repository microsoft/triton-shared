module {
  tt.func @kernel(%arg0: !tt.ptr<bf16, 1>, %arg1: !tt.ptr<bf16, 1>, %arg2: i32) {
    %0 = tt.addptr %arg0, %arg2 : !tt.ptr<bf16, 1>, i32
    %1 = tt.addptr %arg1, %arg2 : !tt.ptr<bf16, 1>, i32
    %2 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : bf16
    tt.store %1, %2 {cache = 1 : i32, evict = 1 : i32} : bf16
    tt.return
  }
}


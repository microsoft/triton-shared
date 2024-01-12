import re

text = """
define void @reduce_kernel_2d_0d1d2de3de(i64 %0, ptr %1, i64 %2, ptr %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11) {
  %13 = insertvalue { i64, ptr } undef, i64 %0, 0
  %14 = insertvalue { i64, ptr } %13, ptr %1, 1
  %15 = insertvalue { i64, ptr } undef, i64 %2, 0
  %16 = insertvalue { i64, ptr } %15, ptr %3, 1
  %17 = mul i32 %4, %9
"""

pattern = r"define void @(\w+)\(.+"
matches = re.findall(pattern, text)

print(matches)

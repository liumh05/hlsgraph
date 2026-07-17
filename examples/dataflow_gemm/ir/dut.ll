; ModuleID = 'dut'
source_filename = "kernel.cpp"

define i32 @compute(i32 %value, i32 %weight) {
entry:
  %mul = mul nsw i32 %value, %weight, !dbg !4
  br label %exit
exit:
  ret i32 %mul
}

!1 = !DIFile(filename: "kernel.cpp", directory: ".")
!4 = !DILocation(line: 18, column: 5, scope: !5)


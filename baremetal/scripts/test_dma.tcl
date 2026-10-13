connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "*A53*#0"}

puts "=== 1. Soft Reset Accelerator ==="
mwr -force 0xA0000000 0x8
after 50
mwr -force 0xA0000000 0x0
mwr -force 0xA000000C 0x7

puts "=== 2. Write Known Pattern to DDR at 0x20000000 ==="
mwr -force 0x20000000 0xDEADBEEF 4
puts "DDR at 0x20000000:"
puts [mrd 0x20000000 4]

puts "=== 3. Clear DDR at 0x38000000 ==="
mwr -force 0x38000000 0x00000000 4
puts "DDR at 0x38000000 before DMA:"
puts [mrd 0x38000000 4]

puts "=== 4. Trigger Read DMA: DDR 0x20000000 -> SRAM Row 10 (1 beat = 16 bytes) ==="
mwr -force 0xA0000010 0x20000000
mwr -force 0xA0000014 0x0000000A
mwr -force 0xA0000018 0x00000001
mwr -force 0xA0000000 0x1
after 100
puts "Status after Read DMA:"
puts [mrd -force 0xA0000004 1]

puts "=== 5. Trigger Write DMA: SRAM Row 10 -> DDR 0x38000000 (1 beat = 16 bytes) ==="
mwr -force 0xA0000020 0x38000000
mwr -force 0xA0000024 0x0000000A
mwr -force 0xA0000028 0x00000001
mwr -force 0xA0000000 0x2
after 100
puts "Status after Write DMA:"
puts [mrd -force 0xA0000004 1]

puts "=== 6. Check DDR at 0x38000000 after Write DMA ==="
puts [mrd 0x38000000 4]

exit


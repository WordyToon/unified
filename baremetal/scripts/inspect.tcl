connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "*A53*#0"}
puts "=== DDR LOGITS AT 0x38000000 ==="
puts [mrd 0x38000000 16]
puts "=== ACCELERATOR REGISTERS 0xA0000000 ==="
puts [mrd -force 0xA0000000 24]
exit

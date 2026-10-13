connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "*A53*#0"}

puts "=== DDR LOGITS AT 0x38000000 RIGHT NOW ==="
puts [mrd 0x38000000 16]

puts "=== RUNNING MNIST (Option 1B) ==="
source c:/Users/m.usman/Documents/cnn_accel/c_engine_architecture/unified/baremetal/scripts/run_xsdb.tcl
# Wait for inference to finish
after 4000

targets -set -filter {name =~ "*A53*#0"}
puts "=== DDR LOGITS AT 0x38000000 AFTER MNIST ==="
puts [mrd 0x38000000 16]

puts "=== ACCELERATOR REGISTERS ==="
puts [mrd -force 0xA0000000 24]

exit


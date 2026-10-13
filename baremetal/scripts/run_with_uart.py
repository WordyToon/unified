#!/usr/bin/env python3
"""
Runs a hardware test script while concurrently capturing and logging UART output from COM7.
"""
import sys
import os
import time
import threading
import subprocess
import serial

PORT = "COM7"
BAUD = 115200

def uart_listener(stop_event, captured_lines):
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.2)
        ser.reset_input_buffer()
        print(f"[*] Serial port {PORT} opened at {BAUD} baud.")
        while not stop_event.is_set():
            data = ser.read(ser.in_waiting or 1)
            if data:
                text = data.decode("latin1", errors="replace")
                captured_lines.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()
        # Read trailing data
        time.sleep(1.0)
        trailing = ser.read(ser.in_waiting or 0)
        if trailing:
            text = trailing.decode("latin1", errors="replace")
            captured_lines.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        ser.close()
    except Exception as e:
        print(f"[-] Serial listener error on {PORT}: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_with_uart.py <command to run...>")
        sys.exit(1)

    cmd = sys.argv[1:]
    captured = []
    stop_event = threading.Event()

    listener_thread = threading.Thread(target=uart_listener, args=(stop_event, captured), daemon=True)
    listener_thread.start()
    time.sleep(0.5)

    print(f"\n[*] Executing command: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd)

    # Let UART finish printing
    time.sleep(3.0)
    stop_event.set()
    listener_thread.join(timeout=3.0)

    full_output = "".join(captured)
    print("\n" + "=" * 80)
    print(f"  TOTAL CAPTURED UART BYTES FROM {PORT}: {len(full_output)}")
    print("=" * 80)

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_uart_capture.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(full_output)
    print(f"[+] Saved UART capture log to: {log_path}")

    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()


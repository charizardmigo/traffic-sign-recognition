"""
=============================================
 Authors  : Priestley F. and Cynthia W.
 Filename : diagnose_ckpt.py
 Purpose  : Verifies health of checkpoints
 Date     : 17.09.2025
 ============================================
"""

import os
import sys
import torch

path = "workspace/nanodet-plus-m-1.5x_320/model_best/model_best.ckpt"

if not os.path.exists(path):
    print("File not found:", path)
    sys.exit(1)

print("File size:", os.path.getsize(path), "bytes")

# Prints first bytes
with open(path, "rb") as f:
    header = f.read(64)
print("Header (raw):", header)
try:
    print("Header (utf-8):", header.decode("utf-8", errors="replace"))
except Exception:
    pass

# Tries detecting if it's zip (new PyTorch), gzip, or text
if header.startswith(b"PK"):
    print("Detected ZIP (PK) header - likely new torch zip serialization.")
elif header.startswith(b"\x1f\x8b"):
    print("Detected GZIP header.")
elif all(32 <= b < 127 or b in (9,10,13) for b in header):
    print("Header looks mostly printable - may be text (not a binary pickle).")
else:
    print("Header looks binary but not PK/gzip/text.")

# Tries torch.load in a guarded try-except (DO NOT run this if file is untrusted)
print("\nAttempting torch.load (read-only) ...")
try:
    ckpt = torch.load(path, map_location="cpu")
    print("torch.load succeeded. Type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("top-level keys:", list(ckpt.keys())[:40])
except Exception as e:
    print("torch.load failed:", repr(e))

# Tries to open as zip to list members (if zip)
if header.startswith(b"PK"):
    import zipfile
    try:
        with zipfile.ZipFile(path, 'r') as z:
            print("Zip file members:", z.namelist()[:40])
    except Exception as e:
        print("zipfile read failed:", e)
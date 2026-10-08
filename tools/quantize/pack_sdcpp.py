#!/usr/bin/env python3
# Pack a .safetensors checkpoint into .sdcpp: rowwise int8 weights + f32 scales.
#
# .sdcpp v1 layout (ALL LITTLE-ENDIAN):
#   b'SDCP' | u32 version=1 | u32 tensor_count
#   per tensor: u32 name_len | name bytes | u8 dtype (0=int8 rowwise, 1=f32 raw)
#             | u32 ndim | i64 dims[ndim] | u64 data_offset (into data blob)
#   data blob (starts right after the last entry):
#     dtype 0 -> i8 weights [numel] then f32 scales [rows],
#                rows = numel/dims[-1]; scale[j] = amax(row j)/127 dequantizes row j
#     dtype 1 -> f32 [numel] raw (small or --dtype-preserve tensors)
# Tensors with numel < 65536 are kept raw f32; everything float else is quantized
# along the last dim. Single streaming pass, chunked (peak RAM ~100MB).
import argparse
import json
import os
import struct
import sys

import numpy as np

MAGIC, VERSION = b"SDCP", 1
MIN_QUANT_NUMEL = 65536
CHUNK_ELEMS = 4_000_000  # rows per chunk ~= this many elements (16MB f32)

# source dtype -> (numpy dtype, elemsize, is_float)
SRC = {
    "BF16": (np.uint16, 2, True),
    "F16": (np.float16, 2, True),
    "F32": (np.float32, 4, True),
    "I8": (np.int8, 1, False),
    "U8": (np.uint8, 1, False),
    "I32": (np.int32, 4, False),
    "I64": (np.int64, 8, False),
}


def to_f32(raw, dt):
    a = np.frombuffer(raw, dtype=SRC[dt][0])
    if dt == "BF16":
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def quant_rows(x):
    # x [m,last] f32 -> (i8 [m,last], f32 scale [m]); scale = amax/127
    amax = np.abs(x).max(axis=1)
    safe = np.where(amax > 0, amax, 1.0)
    q = np.rint(x * (127.0 / safe)[:, None])
    np.clip(q, -127, 127, out=q)
    return q.astype(np.int8), (amax / 127.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="pack .safetensors -> .sdcpp (rowwise int8)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--only", default="", help="comma-separated name prefixes to include")
    ap.add_argument("--dtype-preserve", default="",
                    help="colon-separated source dtypes kept raw f32, e.g. f32:i8")
    a = ap.parse_args()
    only = [p for p in a.only.split(",") if p]
    preserve_dt = set(p for p in a.dtype_preserve.split(":") if p)

    f = open(a.input, "rb")
    nlen = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(nlen))
    data_base = 8 + nlen

    tens = [(n, e) for n, e in hdr.items()
            if n != "__metadata__"
            and (not only or any(n.startswith(p) for p in only))]
    tens.sort(key=lambda t: t[1]["data_offsets"][0])  # sequential read order

    # pass 1: header math only — decide quantize vs preserve, assign blob offsets
    plan, off = [], 0
    for name, e in tens:
        dims, numel = e["shape"], 1
        for d in dims:
            numel *= d
        dt = e["dtype"]
        if dt not in SRC:
            sys.exit("unsupported dtype %s for %s" % (dt, name))
        quant = SRC[dt][2] and dt not in preserve_dt and numel >= MIN_QUANT_NUMEL
        rows = numel // dims[-1] if (quant and dims and dims[-1]) else 0
        blob = numel + rows * 4 if quant else numel * 4
        plan.append((name, 0 if quant else 1, dims, off, numel, rows))
        off += blob

    with open(a.output, "wb") as out:
        out.write(MAGIC + struct.pack("<II", VERSION, len(plan)))
        for name, code, dims, o, numel, rows in plan:
            nb = name.encode()
            out.write(struct.pack("<I", len(nb)) + nb + struct.pack("<B", code))
            out.write(struct.pack("<I", len(dims)))
            out.write(struct.pack("<%dq" % len(dims), *dims))
            out.write(struct.pack("<Q", o))
        # pass 2: stream tensor data (weights chunks, then the tensor's scales)
        for (name, code, dims, o, numel, rows), (_, e) in zip(plan, tens):
            beg, _ = e["data_offsets"]
            dt = e["dtype"]
            _, esz, _ = SRC[dt]
            last = dims[-1] if dims and dims[-1] else 1
            nrows = (numel + last - 1) // last
            scales = np.zeros(nrows, np.float32) if code == 0 else None
            step = max(1, CHUNK_ELEMS // last)
            for r0 in range(0, nrows, step):
                r1 = min(nrows, r0 + step)
                # ponytail: os.pread not mmap — mmap pages pile up in RSS on a 23GB read
                raw = os.pread(f.fileno(), (r1 - r0) * last * esz,
                               data_base + beg + r0 * last * esz)
                x = to_f32(raw, dt)
                if code == 0:
                    q, s = quant_rows(x.reshape(r1 - r0, last))
                    out.write(q.tobytes())
                    scales[r0:r1] = s
                else:
                    out.write(x.astype(np.float32, copy=False).tobytes())
            if code == 0:
                out.write(scales.tobytes())
            print("%-55s %-5s %-22s %d bytes" %
                  (name, "i8" if code == 0 else "f32", str(dims),
                   numel + rows * 4 if code == 0 else numel * 4))
    print("packed %d tensors, out=%.2f GB" % (len(plan), off / 1e9))


if __name__ == "__main__":
    main()

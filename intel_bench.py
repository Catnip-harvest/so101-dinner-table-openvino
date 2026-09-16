"""
Intel Core Ultra benchmark for the lablab Physical AI Challenge.  Windows 11, pure Python.
Deliberately has NO torch and NO LeRobot dependency - this is the deploy-side bundle.

    pip install openvino numpy
    python intel_bench.py                       # stage 1: what devices does this box expose?
    python intel_bench.py --ir policy/model.xml # stage 2: latency per device and precision

Stage 1 takes seconds, needs nothing but OpenVINO, and is worth running the moment the box
is reachable: it produces the device list that the README and the benchmark table are built
on, and it proves the NPU driver is alive before anything depends on it.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

try:
    import openvino as ov
except ImportError:
    sys.exit("openvino is not installed.  pip install openvino numpy")


def probe():
    """List every inference device this machine exposes, with its full product name."""
    core = ov.Core()
    print("OpenVINO", ov.__version__)
    rows = []
    for name in core.available_devices:
        try:
            full = core.get_property(name, "FULL_DEVICE_NAME")
        except Exception as exc:
            full = "<unreadable: %s>" % type(exc).__name__
        rows.append({"device": name, "full_name": str(full)})
        print("  %-8s %s" % (name, full))
    if not any(d.startswith("NPU") for d in core.available_devices):
        print("  NOTE: no NPU listed.  Either this box has none, or - on Panther Lake -")
        print("        the driver is not installed.  CPU and GPU benchmark fine without it.")
    return rows


def _fill(port):
    """Build a plausible tensor for one input, pinning any dynamic dimension to 1.

    Booleans are filled with True, not zeros: the boolean inputs on a VLA graph are
    attention and camera-presence masks, and an all-False mask makes the model emit
    NaN.  The graph is static so the values do not change the latency, but a NaN
    result would hide a broken export behind a perfectly healthy-looking number.
    """
    shape = []
    for dim in port.get_partial_shape():
        shape.append(1 if dim.is_dynamic else dim.get_length())
    dtype = port.get_element_type().to_dtype()
    if np.issubdtype(dtype, np.floating):
        return np.random.rand(*shape).astype(dtype)
    if dtype == np.bool_:
        return np.ones(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def bench(ir_path, device, runs=50, warmup=10):
    """Compile the IR on one device and time single-frame inference."""
    core = ov.Core()
    model = core.read_model(ir_path)
    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    compile_s = time.perf_counter() - t0

    request = compiled.create_infer_request()
    feed = {p.get_any_name(): _fill(p) for p in compiled.inputs}

    result = {}
    for _ in range(max(1, warmup)):
        result = request.infer(feed)

    # A latency number means nothing if the graph is producing NaN.  Check once.
    finite = all(bool(np.isfinite(np.asarray(v)).all()) for v in result.values())

    samples = []
    for _ in range(runs):
        t = time.perf_counter()
        request.infer(feed)
        samples.append((time.perf_counter() - t) * 1000.0)

    samples.sort()
    # Nearest-rank p95.  The obvious `int(n * 0.95) - 1` is off by one and can land
    # below the mean on small run counts - at runs=3 it returns the median.
    p95_index = -(-95 * len(samples) // 100) - 1
    return {
        "device": device,
        "compile_s": round(compile_s, 2),
        "mean_ms": round(statistics.mean(samples), 2),
        "p50_ms": round(samples[len(samples) // 2], 2),
        "p95_ms": round(samples[p95_index], 2),
        "fps": round(1000.0 / statistics.mean(samples), 1),
        "runs": runs,
        "output_finite": finite,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", help="path to an OpenVINO .xml; omit to only probe devices")
    ap.add_argument("--devices", default="", help="comma list, e.g. CPU,GPU,NPU (default: all)")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    result = {"devices": probe(), "benchmarks": []}

    if args.ir:
        wanted = [d.strip() for d in args.devices.split(",") if d.strip()]
        wanted = wanted or ov.Core().available_devices
        # Record which IR these numbers belong to.  Three precisions of the same policy
        # produce three near-identical JSON files otherwise.
        weights = os.path.splitext(args.ir)[0] + ".bin"
        result["ir"] = os.path.abspath(args.ir)
        result["ir_weights_mb"] = (round(os.path.getsize(weights) / 1e6, 1)
                                   if os.path.exists(weights) else None)
        print("\nbenchmarking", args.ir, "(%s MB of weights)" % result["ir_weights_mb"])
        for device in wanted:
            try:
                row = bench(args.ir, device, args.runs)
                result["benchmarks"].append(row)
                print("  %-8s %7.2f ms mean | %7.2f p95 | %6.1f fps | compile %.2fs%s"
                      % (row["device"], row["mean_ms"], row["p95_ms"], row["fps"],
                         row["compile_s"], "" if row["output_finite"] else "  OUTPUT NOT FINITE"))
            except Exception as exc:
                print("  %-8s FAILED: %s" % (device, str(exc)[:120]))
                result["benchmarks"].append({"device": device, "error": str(exc)[:300]})

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

"""遥测覆盖和 profiler trace 的离线回归。"""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from hicache.prepare import write_json
from tools.diagnostics import (EnvironmentMonitor, merge_intervals, overlap_duration,
                               sample_gpu_devices, trace_intervals)


class Diagnostics(unittest.TestCase):
    def test_trace_correlates_kernels_and_counts_overlaps_once(self):
        self.assertEqual(merge_intervals([(1, 4), (2, 3), (4, 6), (8, 9)]), [(1, 6), (8, 9)])
        self.assertEqual(overlap_duration([(1, 6), (8, 9)], [(3, 8.5)]), 3.5)
        trace = [
            dict(cat="user_annotation", name="step[EXTEND bs=1 toks=128]", pid=1, tid=1, ts=100, dur=100),
            dict(cat="cuda_runtime", pid=1, tid=1, ts=110, dur=2, args=dict(correlation=7)),
            dict(cat="cuda_runtime", pid=1, tid=2, ts=110, dur=2, args=dict(correlation=9)),
            dict(cat="kernel", name="gemm", ts=150, dur=30, args=dict(device=0, correlation=7)),
            dict(cat="kernel", name="hicache_transfer_per_layer", ts=140, dur=30, args=dict(device=0, correlation=8)),
            dict(cat="kernel", name="hicache_relayout_kernel", ts=190, dur=20, args=dict(device=0, correlation=9)),
        ]
        forward, load, details = trace_intervals(trace)
        self.assertEqual((forward, load), ([(150, 180)], [(140, 170)]))
        self.assertEqual(details["forward_input_tokens"], 128)
        self.assertEqual(overlap_duration(forward, load), 20)

    def test_nvml_failure_is_local_to_the_failed_device(self):
        def handle(index):
            if index == 0:
                raise RuntimeError("GPU0 lost")
            return index

        nvml = SimpleNamespace(
            NVMLError=RuntimeError, NVML_TEMPERATURE_GPU=0, NVML_CLOCK_SM=0, NVML_CLOCK_MEM=1,
            nvmlDeviceGetCount=lambda: 4, nvmlDeviceGetHandleByIndex=handle,
            nvmlDeviceGetUtilizationRates=lambda _: SimpleNamespace(gpu=0, memory=0),
            nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(used=1, total=48),
            nvmlDeviceGetComputeRunningProcesses=lambda _: [SimpleNamespace(pid=42, usedGpuMemory=1)],
            nvmlDeviceGetUUID=lambda index: f"GPU-{index}",
        )
        sampled = sample_gpu_devices(nvml, 3)
        self.assertTrue(sampled["ok"])
        self.assertFalse(sampled["gpus"]["0"]["ok"])
        self.assertEqual(sampled["gpus"]["3"]["compute_processes"], [dict(pid=42, used_bytes=1)])
        for index in (0, 9):
            self.assertFalse(sample_gpu_devices(nvml, index)["ok"])


class Telemetry(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        write_json(self.root / "owner.json", dict(gpu_index=3, allowed_gpu_pids=[42], scheduler_pid=42))
        self.start = time.monotonic_ns() - 10_000_000_000

    def write_samples(self, kind, seconds=(0, 1, 2, 3), *, ok=True, pid=42):
        with (self.root / f"{kind}.jsonl").open("w") as stream:
            for offset in seconds:
                record = dict(ok=ok, duration_ms=2, end_monotonic_ns=self.start + offset * 10**9,
                              gpus={"3": dict(compute_processes=[dict(pid=pid)])})
                print(json.dumps(record), file=stream)

    def test_window_rejects_extra_process_failed_sampler_or_gap(self):
        window = dict(measurement_start_monotonic_ns=self.start + 1_200_000_000,
                      measurement_end_monotonic_ns=self.start + 1_400_000_000)
        for gpu, host, valid in [({}, {}, True), ({"pid": 99}, {}, False),
                                 ({"ok": False}, {}, False), ({}, {"ok": False}, False),
                                 ({"seconds": (0, 4)}, {}, False)]:
            with self.subTest(gpu=gpu, host=host):
                self.write_samples("gpu", **gpu)
                self.write_samples("host", **host)
                self.assertEqual(EnvironmentMonitor(self.root).evaluate(window)["environment_valid"], valid)

    def test_incomplete_json_line_is_read_once_when_finished(self):
        monitor = EnvironmentMonitor(self.root)
        with (self.root / "gpu.jsonl").open("w", buffering=1) as stream:
            stream.write('{"end_monotonic_ns":')
            stream.flush()
            monitor.refresh()
            self.assertEqual(monitor.records["gpu"], [])
            stream.write(str(self.start) + "}\n")
            monitor.refresh()
            monitor.refresh()
            self.assertEqual(monitor.records["gpu"], [{"end_monotonic_ns": self.start}])

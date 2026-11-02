"""无 GPU 回归：真实脚本/HTTP/进程边界，服务与 Docker 使用替身。"""

import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class Launchers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hicache launch ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model with spaces"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        self.source = self.root / "source with spaces"
        package = self.source / "python/sglang"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "launch_server.py").write_text('''
import json, os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, lambda *_: sys.exit(23))
Path(os.environ['FAKE_CAPTURE']).write_text(json.dumps({
    'args': sys.argv[1:], 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'timer': os.environ.get('SGLANG_ENABLE_METRICS_DEVICE_TIMER'),
    'tree': os.environ.get('SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND')}))
print('fake server started', flush=True)
if os.environ.get('FAKE_WAIT'):
    while True: time.sleep(.05)
sys.exit(int(os.environ.get('FAKE_EXIT', '0')))
''')
        self.bin = self.root / "bin"
        self.bin.mkdir()
        smi = self.bin / "nvidia-smi"
        smi.write_text("#!/bin/sh\nprintf 'test GPU, no real device access\\n'\n")
        smi.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if k not in (
            "MODEL_ROOT",
            "GPU", "HOST", "PORT", "HICACHE", "HICACHE_SIZE", "FORWARD_TIMING", "TP", "DTYPE",
            "KV_DTYPE", "CONTEXT_LENGTH", "MAX_TOTAL_TOKENS", "MEM_FRACTION", "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES", "SGLANG_ENABLE_METRICS_DEVICE_TIMER", "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND",
        )}
        self.env.update(PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        MODEL_DIR=str(self.model), SGLANG_DIR=str(self.source),
                        PYTHON_BIN=sys.executable, RESULTS_DIR=str(self.root / "results"),
                        RUN_ID="test", FAKE_CAPTURE=str(self.root / "capture.json"))

    def run_script(self, script, *args, **env):
        return subprocess.run(["bash", str(PROJECT / script), *args], env={**self.env, **env},
                              text=True, capture_output=True, timeout=15)

    def test_native_records_actual_command_and_failure(self):
        result = self.run_script("serve.sh", "--json-model-override-args", '{"max_position_embeddings":135168}',
                                 FAKE_EXIT="7", GPU="2", HICACHE="1", FORWARD_TIMING="1")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("fake server started", result.stdout)
        capture = json.loads((self.root / "capture.json").read_text())
        self.assertEqual((capture["gpu"], capture["timer"], capture["tree"]), ("2", "1", "python"))
        self.assertEqual(capture["args"][-1], '{"max_position_embeddings":135168}')
        manifest = json.loads((self.root / "results/test/launch.json").read_text())
        self.assertEqual(manifest["runtime"], "native")
        self.assertEqual(manifest["command"][3:], capture["args"])
        self.assertTrue((self.root / "results/test/launch-source/serve.sh").exists())
        # 重复启动不得覆盖记录，也不应再次运行模型进程。
        (self.root / "capture.json").unlink()
        self.assertNotEqual(self.run_script("serve.sh").returncode, 0)
        self.assertFalse((self.root / "capture.json").exists())

    def test_installed_package_and_inherited_gpu(self):
        result = self.run_script("serve.sh", SGLANG_DIR="", PYTHONPATH=str(self.source / "python"),
                                 CUDA_VISIBLE_DEVICES="GPU-already-selected")
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.root / "results/test/launch.json").read_text())
        self.assertIsNone(manifest["sglang_source"])
        self.assertFalse(manifest["source_version_available"])
        capture = json.loads((self.root / "capture.json").read_text())
        self.assertEqual(capture["gpu"], "GPU-already-selected")

    def test_native_forwards_termination_via_exec(self):
        process = subprocess.Popen(["bash", str(PROJECT / "serve.sh")],
                                   env={**self.env, "FAKE_WAIT": "1"},
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while not (self.root / "capture.json").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    self.fail("fake server did not start")
                time.sleep(.03)
            process.send_signal(signal.SIGTERM)
            output, _ = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 23, output)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)

    def test_docker_and_native_share_server_arguments(self):
        overrides = dict(GPU="3", HICACHE="1", FORWARD_TIMING="1", CONTEXT_LENGTH="135168",
                         MAX_TOTAL_TOKENS="135168", HICACHE_SIZE="32", MEM_FRACTION="0.85")
        extra = ["--json-model-override-args", '{"max_position_embeddings":135168}']
        docker = self.run_script("launch_server.sh", "--dry-run", *extra, **overrides)
        self.assertEqual(docker.returncode, 0, docker.stderr)
        tokens = shlex.split(docker.stdout)
        container_env = {}
        for i, token in enumerate(tokens):
            if token == "--env":
                key, value = tokens[i + 1].split("=", 1)
                container_env[key] = value
        self.assertNotIn("CUDA_VISIBLE_DEVICES", container_env)
        self.assertIn("device=3", tokens)
        self.assertEqual(tokens[tokens.index("bash"):], ["bash", "/experiment/serve.sh", "--exec", *extra])
        # 模拟 Docker 的变量传递，在同一个 serve.sh 中得到与 native 相同的参数。
        via_docker = self.run_script("serve.sh", "--dry-run", *extra, **container_env)
        native = self.run_script("serve.sh", "--dry-run", *extra, **overrides)
        a, b = shlex.split(via_docker.stdout), shlex.split(native.stdout)
        a, b = a[a.index("-m"):], b[b.index("-m"):]
        a[a.index("--model-path") + 1] = str(self.model)
        self.assertEqual(a, b)
        self.assertFalse((self.root / "results").exists())

    def fake_docker(self):
        docker = self.bin / "docker"
        docker.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['FAKE_DOCKER_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
command = sys.argv[1:]
if command[:2] == ['image', 'inspect']: print('sha256:fixed-image')
elif command[0] == 'run': print('owned-container-id')
elif command[0] == 'logs': print('fake container exited')
elif command[0] == 'wait': print('17')
''')
        docker.chmod(0o755)
        log = self.root / "docker.jsonl"
        return log

    def test_docker_propagates_exit_and_cleans_only_owned_container(self):
        log = self.fake_docker()
        result = self.run_script("launch_server.sh", FAKE_DOCKER_LOG=str(log))
        self.assertEqual(result.returncode, 17, result.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(calls[-2:], [["stop", "--time", "15", "owned-container-id"], ["rm", "owned-container-id"]])
        manifest = json.loads((self.root / "results/test/launch.json").read_text())
        self.assertEqual(manifest["image_id"], "sha256:fixed-image")
        self.assertNotIn("packages", manifest)

    def test_named_native_profile_snapshot_and_cli_override(self):
        literal = json.dumps({"text": "$(touch SHOULD_NOT_EXIST); `echo nope`", "max_position_embeddings": 4096})
        result = self.run_script("serve.sh", "qwen3-8b-4k", "--mem-fraction-static", "0.85",
                                 "--json-model-override-args", literal, MEM_FRACTION="0.75")
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = self.root / "results/test"
        capture = json.loads((self.root / "capture.json").read_text())
        self.assertEqual(capture["timer"], "1")
        self.assertEqual(capture["args"], ["--config", str(directory / "server-config.yaml"),
                                        "--mem-fraction-static", "0.85", "--json-model-override-args", literal])
        resolution = json.loads((directory / "profile-resolution.json").read_text())
        self.assertEqual(resolution["server"]["context-length"], 4096)
        self.assertEqual(resolution["server"]["mem-fraction-static"], .75)
        self.assertEqual(resolution["server"]["model-path"], str(self.model))
        manifest = json.loads((directory / "launch.json").read_text())
        raw = (PROJECT / "configs/serve/qwen3-8b-4k.yaml").read_bytes()
        self.assertEqual((directory / "profile.yaml").read_bytes(), raw)
        self.assertEqual(manifest["profile"]["source_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(manifest["profile"]["server_config_sha256"],
                         hashlib.sha256((directory / "server-config.yaml").read_bytes()).hexdigest())
        # 已冻结的配置不再次读取当下的环境覆盖。
        frozen = self.run_script("serve.sh", "--exec", "--resolved-profile",
                                 str(directory / "profile-resolution.json"), HICACHE="0", FORWARD_TIMING="0")
        self.assertEqual(frozen.returncode, 0, frozen.stderr)
        self.assertEqual(json.loads((self.root / "capture.json").read_text())["timer"], "1")
        self.assertFalse((PROJECT / "SHOULD_NOT_EXIST").exists())

    def test_named_docker_profile_freezes_config_and_rewrites_model(self):
        log = self.fake_docker()
        result = self.run_script("launch_server.sh", "qwen3-8b-4k", "--model", str(self.model),
                                 "--mem-fraction-static", "0.85", FAKE_DOCKER_LOG=str(log))
        self.assertEqual(result.returncode, 17, result.stderr)
        directory = self.root / "results/test"
        resolution = json.loads((directory / "profile-resolution.json").read_text())
        self.assertEqual(resolution["server"]["model-path"], "/model")
        self.assertEqual(resolution["cli_args"], ["--mem-fraction-static", "0.85"])
        self.assertEqual(resolution["model_directory"], str(self.model))
        manifest = json.loads((directory / "launch.json").read_text())
        self.assertEqual(manifest["command"][-2:], ["--resolved-profile", "/results/test/profile-resolution.json"])
        self.assertIn(f"type=bind,src={self.model},dst=/model,readonly", manifest["command"])

    def test_profile_listing_and_preview_need_no_model_or_gpu(self):
        for script in ("serve.sh", "launch_server.sh"):
            listed = self.run_script(script, "--list-configs", MODEL_DIR="")
            self.assertEqual(listed.returncode, 0, listed.stderr)
            self.assertIn("qwen3-8b-4k", listed.stdout)
            for args in (["qwen3-8b-4k", "--dry-run"], ["--dry-run", "qwen3-8b-4k"]):
                result = self.run_script(script, *args, MODEL_DIR="", MODEL_ROOT="/not downloaded yet")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("context-length: 4096", result.stdout)
                self.assertIn("/not downloaded yet/Qwen3-8B", result.stdout)
        self.assertFalse((self.root / "results").exists())

    def test_bad_profile_fails_before_docker_or_server(self):
        log = self.fake_docker()
        for script in ("serve.sh", "launch_server.sh"):
            bad_name = self.run_script(script, "not-an-existing-profile", FAKE_DOCKER_LOG=str(log))
            self.assertNotEqual(bad_name.returncode, 0)
            self.assertIn("--list-configs", bad_name.stderr)
            missing = self.run_script(script, "qwen3-8b-4k", MODEL_DIR="", MODEL_ROOT="", FAKE_DOCKER_LOG=str(log))
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("MODEL_ROOT", missing.stderr)
        self.assertFalse(log.exists())
        self.assertFalse((self.root / "capture.json").exists())


class RemoteClient(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hicache client ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # 只复制客户端，不提供 serve.sh、Docker、SGLang 或模型权重。
        shutil.copyfile(PROJECT / "run_experiment.py", self.root / "run_experiment.py")
        (self.root / "transformers.py").write_text('''
class AutoTokenizer:
    all_special_ids = []
    @classmethod
    def from_pretrained(cls, path, **kwargs): return cls()
    def encode(self, text, **kwargs): return list(text.encode())
''')
        for name in ("torch", "sglang"):
            (self.root / f"{name}.py").write_text("raise AssertionError('client must not import server dependencies')\n")
        self.requests = []
        cache = []
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass

            def do_GET(self): self.handle_request()
            def do_POST(self): self.handle_request()

            def handle_request(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                payload = json.loads(body) if body else None
                requests.append((self.path, payload))
                result = {}
                if self.path == "/server_info":
                    result = dict(enable_metrics=True, disable_radix_cache=False,
                                  enable_hierarchical_cache=False, tp_size=1, pp_size=1, dp_size=1,
                                  max_running_requests=1, context_length=1024, page_size=1,
                                  dtype="bfloat16", kv_cache_dtype="bfloat16")
                elif self.path == "/model_info":
                    result = {"model_path": "/server-only/model-not-on-client"}
                elif self.path.startswith("/flush_cache"):
                    cache.clear()
                elif self.path == "/generate":
                    ids = payload["input_ids"]
                    hit = 0
                    for a, b in zip(cache, ids):
                        if a != b: break
                        hit += 1
                    cache[:] = ids
                    result = dict(output_ids=[7], meta_info=dict(
                        prompt_tokens=len(ids), completion_tokens=1, cached_tokens=hit,
                        cached_tokens_details=dict(device=hit, host=0, storage=0),
                        num_retractions=0, finish_reason={"type": "length"},
                        e2e_latency=.002, first_token_latency=.001, queue_time=0))
                encoded = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def run_client(self, *extra):
        env = dict(os.environ)
        env.pop("EXPERIMENT_LAUNCH_DIR", None)
        return subprocess.run([
            sys.executable, str(self.root / "run_experiment.py"), "run",
            "--base-url", f"http://127.0.0.1:{self.server.server_port}",
            "--tokenizer", str(self.root), "--output-dir", str(self.root / "results"),
            "--prefix-lengths", "8", "16", "--suffix-lengths", "2", "4",
            "--groups", "cold", "l1", "--repeats", "2", "--no-plot", *extra,
        ], env=env, capture_output=True, text=True, timeout=15)

    def test_standalone_http_matrix_without_server_files(self):
        result = self.run_client()
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads((self.root / "results/config.json").read_text())
        self.assertEqual(config["status"], "complete")
        self.assertEqual(config["packages_scope"], "client")
        self.assertIsNone(config["server_artifacts"])
        with (self.root / "results/samples.csv").open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(row["valid"] == "True" for row in rows))
        self.assertEqual([p.name for p in (self.root / "results/source").iterdir()], ["run_experiment.py"])

    def test_profile_path_belongs_to_server_and_artifacts_are_optional_copies(self):
        artifacts = self.root / "copied-launch"
        artifacts.mkdir()
        (artifacts / "launch.json").write_text('{"runtime":"native","sglang_commit":"test"}')
        (artifacts / "server.log").write_text("copied server log\n")
        result = self.run_client("--profile", "--server-profile-dir", "/server-only/traces",
                                 "--server-artifacts", str(artifacts))
        self.assertEqual(result.returncode, 0, result.stderr)
        profiles = [body for path, body in self.requests if path == "/start_profile"]
        self.assertEqual(len(profiles), 16)
        self.assertTrue(all(body["output_dir"] == "/server-only/traces" for body in profiles))
        self.assertFalse((self.root / "results/traces").exists())
        self.assertTrue((self.root / "results/launch.json").is_file())
        self.assertEqual((self.root / "results/server.log").read_text(), "copied server log\n")

    def test_profile_requires_explicit_server_directory(self):
        result = self.run_client("--profile")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--server-profile-dir", result.stderr)
        self.assertFalse(self.requests)

    def test_telemetry_rejects_unconfirmed_remote_clock(self):
        result = self.run_client("--telemetry-dir", str(self.root / "telemetry"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--telemetry-same-host", result.stderr)
        self.assertFalse(self.requests)
        self.assertEqual(json.loads((self.root / "results/config.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()

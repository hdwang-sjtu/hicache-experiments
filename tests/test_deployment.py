"""无 GPU 回归：真实脚本/HTTP/进程边界，服务与 Docker 使用替身。"""

import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from hicache.metadata import code_files
DRIVER_FILES = code_files()


class Launchers(unittest.TestCase):
    def test_frozen_launch_needs_no_profiles_and_reuses_code_archive(self):
        from hicache.runner import CODE_FILES
        from hicache import config as serve_config
        task = self.root / 'managed-result'
        code = task / 'code'
        code.mkdir(parents=True)
        for name in CODE_FILES:
            (code / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, code / name)
        resolution = serve_config.resolve('qwen3-8b-smoke', 'native', [], self.env)
        resolution['server']['context-length'] = 135168
        resolution['server_yaml'] = serve_config.yaml_module().safe_dump(resolution['server'])
        frozen = task / 'resolved-serve.json'
        frozen.write_text(json.dumps(resolution))
        result = subprocess.run(['bash', str(code / 'serve.sh'), '--resolved-profile', str(frozen)],
                                env={**self.env, 'RESULTS_DIR': str(task), 'RUN_ID': 'server',
                                     'CONTEXT_LENGTH': '1024', 'MODEL_DIR': '/wrong', 'FORWARD_TIMING': '0'},
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        recorded = json.loads((task / 'server/profile-resolution.json').read_text())
        self.assertEqual(recorded['server']['context-length'], 135168)
        self.assertEqual(recorded['server']['model-path'], str(self.model))
        self.assertEqual(recorded['environment']['SGLANG_ENABLE_METRICS_DEVICE_TIMER'], '1')
        self.assertFalse((task / 'server/launch-source').exists())
        self.assertEqual(json.loads((task / 'server/launch.json').read_text())['launcher_source'], '../code')

    def test_missing_profile_fails_before_creating_results(self):
        for script in ('serve.sh', 'launch_server.sh'):
            result = self.run_script(script)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--list-configs', result.stderr)
            self.assertFalse((self.root / 'results').exists())

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
        result = self.run_script("serve.sh", "qwen3-8b-smoke", "--json-model-override-args", '{"max_position_embeddings":135168}',
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
        self.assertNotEqual(self.run_script("serve.sh", "qwen3-8b-smoke").returncode, 0)
        self.assertFalse((self.root / "capture.json").exists())


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


    def test_named_native_profile_snapshot_and_cli_override(self):
        literal = json.dumps({"text": "$(touch SHOULD_NOT_EXIST); `echo nope`", "max_position_embeddings": 4096})
        result = self.run_script("serve.sh", "qwen3-8b-smoke", "--mem-fraction-static", "0.85",
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
        raw = (PROJECT / "configs/experiments/qwen3-8b-smoke.yaml").read_bytes()
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
        result = self.run_script("launch_server.sh", "qwen3-8b-smoke", "--model", str(self.model),
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
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(calls[-2:], [["stop", "--time", "15", "owned-container-id"], ["rm", "owned-container-id"]])

    def test_docker_hf_snapshot_keeps_blobs_and_config_paths(self):
        revision = 'd'*40
        for shared in (False, True):
            with self.subTest(shared=shared):
                cache = self.root/('shared' if shared else 'legacy')/'hub'
                repository = cache/'models--org--model'
                snapshot = repository/'snapshots'/revision
                blobs = repository/'blobs'
                snapshot.mkdir(parents=True)
                blobs.mkdir()
                (blobs/'config').write_text('{}')
                (snapshot/'config.json').symlink_to('../../blobs/config')
                if shared:
                    payload = cache/'blobs/aa'/('a'*64)
                    payload.parent.mkdir(parents=True)
                    (blobs/'config').rename(payload)
                    (blobs/'config').symlink_to(os.path.relpath(payload, blobs))
                run_id = 'hf-shared' if shared else 'hf-legacy'
                self.fake_docker()
                log = self.root/(run_id + '.jsonl')
                result = self.run_script('launch_server.sh', 'qwen3-8b-smoke',
                                         MODEL_DIR=str(snapshot), RUN_ID=run_id, FAKE_DOCKER_LOG=str(log))
                self.assertEqual(result.returncode, 17, result.stderr)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                command = next(call for call in calls if call[0] == 'run')
                model_path = f'/model/{repository.name}/snapshots/{revision}'
                self.assertIn(f'type=bind,src={cache},dst=/model,readonly', command)
                self.assertIn(f'MODEL_DIR={model_path}', command)
                resolution = json.loads((self.root/f'results/{run_id}/profile-resolution.json').read_text())
                self.assertEqual(resolution['server']['model-path'], model_path)


class RemoteClient(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hicache client ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # 只复制客户端，不提供 serve.sh、Docker、SGLang 或模型权重。
        for name in DRIVER_FILES:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, self.root / name)
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
            sys.executable, str(getattr(self, 'script', self.root / "run_experiment.py")), "run",
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
        self.assertEqual(sorted(str(p.relative_to(self.root / "results/source")) for p in (self.root / "results/source").rglob("*") if p.is_file() and "__pycache__" not in p.parts), sorted(DRIVER_FILES))

    def test_managed_client_references_existing_evidence_without_copying(self):
        code, server = self.root / 'code', self.root / 'server'
        alias = self.root / 'experiment'
        code.mkdir()
        server.mkdir()
        alias.mkdir()
        for name in DRIVER_FILES:
            (code / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, code / name)
            # 模拟 Docker 的双重 bind mount：路径不同，底层文件身份相同。
            (alias / name).parent.mkdir(parents=True, exist_ok=True)
            (alias / name).hardlink_to(code / name)
        shutil.copyfile(self.root / 'transformers.py', alias / 'transformers.py')
        (server / 'launch.json').write_text('{"runtime":"native"}')
        (server / 'server.log').write_text('one canonical server log\n')
        self.script = alias / 'run_experiment.py'
        result = self.run_client('--server-artifacts', str(server))
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = self.root / 'results'
        config = json.loads((directory / 'config.json').read_text())
        self.assertEqual(config['source_directory'], '../code')
        self.assertEqual(config['server_artifacts'], '../server')
        self.assertEqual(config['server_log_path'], '../server/server.log')
        for name in ('source', 'launch.json', 'server.log', 'all_cache_valid_summary.csv'):
            self.assertFalse((directory / name).exists(), name)

    def test_remote_client_needs_no_launcher_files(self):
        for name in ('serve.sh', 'launch_server.sh', 'run_all.py'):
            (self.root / name).unlink()
        shutil.rmtree(self.root / 'configs')
        shutil.rmtree(self.root / 'patches')
        result = self.run_client()
        self.assertEqual(result.returncode, 0, result.stderr)
        archived = self.root / 'results/source'
        self.assertTrue((archived / 'run_experiment.py').is_file())
        self.assertTrue((archived / 'hicache/benchmark.py').is_file())
        self.assertFalse((archived / 'serve.sh').exists())

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


    def test_telemetry_rejects_unconfirmed_remote_clock(self):
        result = self.run_client("--telemetry-dir", str(self.root / "telemetry"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--telemetry-same-host", result.stderr)
        self.assertFalse(self.requests)
        self.assertEqual(json.loads((self.root / "results/config.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()

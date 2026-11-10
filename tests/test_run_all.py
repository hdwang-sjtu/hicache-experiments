"""一键任务的核心验收：真实子进程/HTTP，模型计算与下载使用小替身。"""

import argparse
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from hicache.prepare import docker_model_mount, inspect_model, prepare_model
from hicache.runner import CODE_FILES, Job, verify_results

# 提供就绪状态和任务身份，供真实总控轮询；不加载模型、不访问 GPU。
SERVER = '''#!/bin/bash
exec "$PYTHON_BIN" - "$@" <<'CHILD'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
mode=os.environ['TEST_MODE']
if mode=='crash': sys.exit(7)
resolution=json.load(open(sys.argv[sys.argv.index('--resolved-profile')+1]))
identity=resolution['cli_args'][1]
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_): pass
    def do_GET(self):
        self.send_response(503 if mode=='timeout' else 200)
        self.end_headers()
        self.wfile.write(json.dumps({'served_model_name':identity}).encode())
HTTPServer(('127.0.0.1',int(os.environ['PORT'])),Handler).serve_forever()
CHILD
'''


class StubJob(Job):
    def prepare(self):
        self.model = self.project / 'mock-model'
        self.phase('preparing_model')
        if self.env['TEST_MODE']=='download-fail':
            raise RuntimeError('simulated download failure')

    def phase(self, name):
        super().phase(name)
        if name=='testing' and self.env['TEST_MODE']=='interrupt':
            os.kill(os.getpid(), signal.SIGTERM)

    def driver_command(self):
        # Driver 自身已有 HTTP/缓存自测；此处只产生它的输出契约。
        program = '''import csv,json,sys
from pathlib import Path
p=Path(sys.argv[1]); p.mkdir()
(p/'partial.txt').write_text('retained')
if sys.argv[2]=='test-fail': sys.exit(9)
for name in ['inputs.json','responses.jsonl','summary.csv']:(p/name).write_text('{}\\n')
(p/'config.json').write_text(json.dumps({'status':'complete'}))
with (p/'samples.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=['prefix_length','suffix_length','group','valid']);w.writeheader()
    w.writerow(dict(prefix_length=2048,suffix_length=128,group='cold',valid=True))
'''
        return [sys.executable, '-c', program, str(self.root/'samples'), self.env['TEST_MODE']]


class TaskFlow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hicache-task-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root/'project'
        self.project.mkdir()
        for name in CODE_FILES:
            (self.project / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT/name, self.project/name)
        for name in ('configs','patches'):
            shutil.copytree(PROJECT/name,self.project/name,dirs_exist_ok=True)
        (self.project/'serve.sh').write_text(SERVER)
        experiment=yaml.safe_load((self.project/'configs/experiments/qwen3-8b-smoke.yaml').read_text())
        experiment['test'].update(groups=['cold'],repeats=1,**{'gpu-timing':False,'no-plot':True,'ready-timeout':1})
        (self.project/'configs/experiments/qwen3-8b-smoke.yaml').write_text(yaml.safe_dump(experiment))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); self.port=sock.getsockname()[1]

    def job(self, mode):
        args=argparse.Namespace(experiment='qwen3-8b-smoke',runtime='native',run_id=mode)
        env=dict(PATH=os.environ['PATH'],PYTHON_BIN=sys.executable,PORT=str(self.port),
                 TEST_MODE=mode,HICACHE_CACHE_DIR=str(self.root/'cache'))
        return StubJob(args,self.project,env)

    def test_success_and_failure_cleanup(self):
        # 六条有不同清理要求的路径共享同一个端到端场景，避免为每个分支搭脚手架。
        for mode,code,phase in [('success',0,None),('download-fail',1,'preparing_model'),
                                ('crash',1,'starting_server'),('timeout',1,'starting_server'),
                                ('test-fail',1,'testing'),('interrupt',143,'testing')]:
            with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                job=self.job(mode)
                self.assertEqual(job.execute(),code)
                state=json.loads((job.root/'run.json').read_text())
                self.assertEqual(state['status'],'complete' if code==0 else ('interrupted' if code==143 else 'failed'))
                self.assertEqual(state.get('failed_phase'),phase)
                self.assertFalse(job.active_processes)
                self.assertTrue(all(p.poll() is not None for p in job.processes))
                if mode=='test-fail': self.assertTrue((job.root/'samples/partial.txt').exists())
                if mode=='success':
                    self.assertEqual(state['validation']['valid_samples'],1)
                    with self.assertRaises(FileExistsError): self.job(mode).execute()
                    # 有文件但少了有效样本，也不能被宣布完成。
                    (job.root/'samples/samples.csv').write_text('prefix_length,suffix_length,group,valid\n')
                    with self.assertRaises(RuntimeError): verify_results(job.root/'samples',job.test)

    def test_occupied_port_is_preserved_and_docker_cleanup_uses_exact_id(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',self.port)); listener.listen()
            job=self.job('occupied')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(job.execute(),1)
            self.assertFalse(job.processes)
            self.assertGreater(listener.fileno(),0)
        job=self.job('cleanup-docker')
        job.args.runtime='docker'
        (job.root/'server').mkdir(parents=True)
        cid='a'*64
        (job.root/'server/container.id').write_text(cid)
        with patch('hicache.runner.subprocess.run') as command:
            command.return_value=argparse.Namespace(returncode=0,stderr='')
            job.cleanup()
        self.assertEqual([call.args[0] for call in command.call_args_list],
                         [['docker','stop','--time','15',cid],['docker','rm','--force',cid]])

    def test_preparation_uses_returned_model_path_for_server_and_driver(self):
        for local in (None, self.root/'custom-model'):
            job = self.job('prepare-local' if local else 'prepare-default')
            job.model = local
            job.root.mkdir(parents=True)
            job.snapshot()
            actual = local or self.root/('hub/models--org--model/snapshots/'+'b'*40)
            def command(argv, **_):
                if '--output' in argv:
                    output = Path(argv[argv.index('--output')+1])
                    output.write_text(json.dumps({'path': str(actual if argv[3]=='model' else self.root/'source')}))
            with patch.object(job, 'command', side_effect=command) as commands:
                Job.prepare(job)
            self.assertEqual(job.model, actual)
            self.assertEqual(job.env['MODEL_DIR'], str(actual))
            model_args = commands.call_args.args[0]
            self.assertEqual('--directory' in model_args, local is not None)
            self.assertEqual(model_args[3], 'model')
            job.args.runtime = 'docker'
            with patch.object(job, 'container_id', return_value='a'*64):
                driver = Job.driver_command(job)
            self.assertEqual(driver[driver.index('--tokenizer')+1], docker_model_mount(actual)[1])


class ModelPreparation(unittest.TestCase):
    revision = 'b'*40

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hicache-model-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def write_model(directory):
        directory.mkdir(parents=True, exist_ok=True)
        files = {'config.json': b'{}', 'tokenizer.json': b'{}',
                 'model.safetensors.index.json': json.dumps({'weight_map': {'x': 'part1.safetensors', 'y': 'part2.safetensors'}}).encode()}
        for name, part in (('x', 'part1.safetensors'), ('y', 'part2.safetensors')):
            header = json.dumps({name: {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
            files[part] = struct.pack('<Q', len(header)) + header + b'1234'
        for name, data in files.items():
            if not (directory/name).is_file():
                (directory/name).write_bytes(data)

    def snapshot(self, cache, shared=()):
        repository = cache/'models--org--model'
        snapshot = repository/'snapshots'/self.revision
        blobs = repository/'blobs'
        self.write_model(blobs)
        snapshot.mkdir(parents=True)
        for file in blobs.iterdir():
            (snapshot/file.name).symlink_to('../../blobs/' + file.name)
        for name in shared:
            blob = blobs/name
            digest = hashlib.sha256(blob.read_bytes()).hexdigest()
            payload = cache/'blobs'/digest[:2]/digest
            payload.parent.mkdir(parents=True, exist_ok=True)
            (cache/'blobs/.huggingface-shared-blobs').write_text('1')
            blob.rename(payload)
            blob.symlink_to(os.path.relpath(payload, blob.parent))
        return snapshot

    def test_only_explicit_model_directory_is_resolved_before_preparation(self):
        args = argparse.Namespace(experiment='qwen35-35b-matrix', runtime='native', run_id='paths')
        job = Job(args, environ={'MODEL_ROOT': '/unused'})
        self.assertIsNone(job.model)
        self.assertNotIn('model-path', job.resolution['server'])
        self.assertNotIn('MODEL_DIR', job.env)
        self.assertEqual(Job(args, environ={'MODEL_DIR': '/from-env'}).model, Path('/from-env'))
        args.model_dir = '/from-cli'
        self.assertEqual(Job(args, environ={'MODEL_DIR': '/from-env'}).model, Path('/from-cli'))

    def test_complete_directory_is_reused_without_downloading_or_writing(self):
        directory = self.root/'local'
        self.write_model(directory)
        before = {p.name: p.stat().st_mtime_ns for p in directory.iterdir()}
        with patch('huggingface_hub.snapshot_download') as download:
            info = prepare_model('org/model', self.revision, directory)
        download.assert_not_called()
        self.assertFalse(info['download_requested'])
        self.assertEqual(info['path'], str(directory))
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in directory.iterdir()})

    def test_dry_run_does_not_prepare_model_cache_or_results(self):
        env = {k: v for k, v in os.environ.items() if k not in ('MODEL_DIR', 'MODEL_ROOT')}
        env.update(HF_HOME=str(self.root/'hf'), HICACHE_CACHE_DIR=str(self.root/'source-cache'),
                   RESULTS_DIR=str(self.root/'results'))
        result = subprocess.run([sys.executable, str(PROJECT/'run_all.py'), 'qwen35-122b-128k', '--dry-run'],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads(result.stdout)
        self.assertIsNone(preview['model_directory'])
        self.assertEqual(preview['model_location'], 'huggingface-default-cache')
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_directory_interruption_and_missing_shard_are_completed(self):
        directory = self.root/'new'
        calls = []
        def download(**kwargs):
            calls.append(kwargs)
            self.write_model(directory)
            if len(calls)==1:
                (directory/'part2.safetensors').unlink()
                raise OSError('interrupted')
            return str(directory)
        with self.assertRaises(OSError):
            prepare_model('org/model', self.revision, directory, download)
        weight = directory/'part1.safetensors'
        unchanged = weight.stat().st_mtime_ns
        info = prepare_model('org/model', self.revision, directory, download)
        self.assertTrue(info['download_requested'])
        self.assertEqual(calls[-1]['local_dir'], str(directory))
        self.assertIn('part1.safetensors', calls[-1]['ignore_patterns'])
        self.assertNotIn('part2.safetensors', calls[-1]['ignore_patterns'])
        self.assertEqual(weight.stat().st_mtime_ns, unchanged)
        self.assertFalse(prepare_model('org/model', self.revision, directory, download)['download_requested'])
        self.assertEqual(len(calls), 2)
        self.assertFalse((directory/'.hicache-model.json').exists())
        self.assertFalse((directory.parent/(directory.name+'.lock')).exists())

    def test_corrupt_files_and_recorded_revision_conflicts_are_not_overwritten(self):
        directory = self.root/'local'
        self.write_model(directory)
        part = directory/'part1.safetensors'
        original = part.read_bytes()
        part.write_bytes(b'broken')
        with patch('huggingface_hub.snapshot_download') as download:
            with self.assertRaises(ValueError): prepare_model('org/model', self.revision, directory)
            download.assert_not_called()
        part.write_bytes(original)
        evidence = [directory/'.hicache-model.json', directory/'.cache/huggingface/download/config.json.metadata']
        for marker in evidence:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({'repo': 'org/other', 'revision': 'c'*40}) if marker.suffix=='.json' else 'c'*40+'\n')
            with patch('huggingface_hub.snapshot_download') as download:
                with self.assertRaises(ValueError): prepare_model('org/model', self.revision, directory)
                download.assert_not_called()
            marker.unlink()

    def test_default_download_uses_only_hf_arguments_and_returned_path(self):
        snapshot = self.snapshot(self.root/'arbitrary-hf-cache')
        with patch('huggingface_hub.snapshot_download', return_value=str(snapshot)) as download:
            info = prepare_model('org/model', self.revision)
        download.assert_called_once_with(repo_id='org/model', revision=self.revision)
        self.assertEqual(info['path'], str(snapshot))
        self.assertEqual(info['revision'], self.revision)
        self.assertEqual(info['source'], 'huggingface')
        self.assertFalse((self.root/'arbitrary-hf-cache/.hicache').exists())
        with patch('huggingface_hub.constants.HF_HUB_OFFLINE', True), \
                patch('huggingface_hub.snapshot_download', return_value=str(snapshot)) as download:
            prepare_model('org/model', self.revision)
        download.assert_called_once_with(repo_id='org/model', revision=self.revision, local_files_only=True)

    def test_explicit_hf_snapshot_repairs_in_hub_cache_and_keeps_symlinks(self):
        cache = self.root/'hub'
        snapshot = self.snapshot(cache)
        blob = snapshot.parent.parent/'blobs/part2.safetensors'
        blob.unlink()
        def download(**kwargs):
            self.assertEqual(kwargs, dict(repo_id='org/model', revision=self.revision, cache_dir=str(cache)))
            self.write_model(blob.parent)
            return str(snapshot)
        info = prepare_model('org/model', self.revision, snapshot, download)
        self.assertEqual(info['path'], str(snapshot))
        self.assertTrue((snapshot/'part2.safetensors').is_symlink())
        with self.assertRaises(ValueError): prepare_model('org/model', 'c'*40, snapshot)
        local = self.root/'local'
        local.mkdir()
        for file in snapshot.iterdir():
            (local/file.name).symlink_to(file.resolve())
        with self.assertRaises(ValueError): inspect_model(local)

    def test_shared_and_mixed_snapshots_reuse_files_and_keep_docker_links(self):
        for shared in (('part1.safetensors',), ('part1.safetensors', 'part2.safetensors')):
            with self.subTest(shared=shared):
                cache = self.root/str(len(shared))/'hub'
                snapshot = self.snapshot(cache, shared=shared)
                before = {p: (p.stat().st_ino, p.stat().st_mtime_ns)
                          for p in cache.rglob('*') if p.is_file()}
                with patch('huggingface_hub.snapshot_download') as download:
                    info = prepare_model('org/model', self.revision, snapshot)
                download.assert_not_called()
                self.assertEqual(info['revision'], self.revision)
                self.assertEqual(before, {p: (p.stat().st_ino, p.stat().st_mtime_ns)
                                         for p in cache.rglob('*') if p.is_file()})
                host, container = docker_model_mount(snapshot)
                self.assertEqual(host, str(cache))
                # 用独立目录模拟挂载后的文件系统；遗漏共享 blobs 会形成断链。
                mounted = self.root/str(len(shared))/'container/model'
                shutil.copytree(host, mounted, symlinks=True)
                model = mounted/Path(container).relative_to('/model')
                self.assertEqual(inspect_model(model)['weight_bytes'], info['weight_bytes'])
                for file in snapshot.iterdir():
                    self.assertEqual((model/file.name).read_bytes(), file.read_bytes())

    def test_shared_blob_missing_file_is_repaired_in_same_cache(self):
        cache = self.root/'hub'
        snapshot = self.snapshot(cache, shared=('part2.safetensors',))
        payload = (snapshot/'part2.safetensors').resolve()
        data = payload.read_bytes()
        payload.unlink()
        def download(**kwargs):
            self.assertEqual(kwargs, dict(repo_id='org/model', revision=self.revision, cache_dir=str(cache)))
            payload.write_bytes(data)
            return str(snapshot)
        info = prepare_model('org/model', self.revision, snapshot, download)
        self.assertTrue(info['download_requested'])
        self.assertTrue((snapshot/'part2.safetensors').is_symlink())

    def test_shared_blobs_cannot_escape_to_other_repositories_or_external_files(self):
        cache = self.root/'hub'
        snapshot = self.snapshot(cache, shared=('part1.safetensors',))
        payload = (snapshot/'part1.safetensors').resolve()
        data = payload.read_bytes()
        for outside in (self.root/'external.safetensors',
                        cache/'models--org--other/blobs/part1.safetensors'):
            with self.subTest(outside=outside):
                outside.parent.mkdir(parents=True, exist_ok=True)
                outside.write_bytes(data)
                payload.unlink()
                payload.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, '目录外'):
                    inspect_model(snapshot)

    def test_real_hf_offline_cache_respects_standard_environment_variables(self):
        clean_env = {k: v for k, v in os.environ.items() if k not in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'XDG_CACHE_HOME')}
        for shared in ((), ('part1.safetensors', 'part2.safetensors')):
            base = self.root/('shared' if shared else 'legacy')
            cache = base/'hub'
            snapshot = self.snapshot(cache, shared=shared)
            for override in ({'HF_HOME': str(base)}, {'HF_HUB_CACHE': str(cache), 'HF_HOME': str(self.root/'other')}):
                with self.subTest(shared=shared, override=override):
                    output = self.root/'model.json'
                    result = subprocess.run([sys.executable, '-m', 'hicache.prepare', 'model', '--load-format', 'auto',
                                             '--repo', 'org/model', '--revision', self.revision, '--output', str(output)],
                                            env={**clean_env, **override, 'HF_HUB_OFFLINE': '1'},
                                            capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(output.read_text())['path'], str(snapshot))


if __name__=='__main__': unittest.main()

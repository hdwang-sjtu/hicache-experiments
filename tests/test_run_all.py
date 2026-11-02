"""一键任务的核心验收：真实子进程/HTTP，模型计算与下载使用小替身。"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from prepare_runtime import prepare_model
from run_all import CODE_FILES, Job, verify_results

# 提供就绪状态和任务身份，供真实总控轮询；不加载模型、不访问 GPU。
SERVER = '''#!/bin/bash
exec "$PYTHON_BIN" - "$@" <<'CHILD'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
mode=os.environ['TEST_MODE']
if mode=='crash': sys.exit(7)
identity=sys.argv[sys.argv.index('--served-model-name')+1]
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
            shutil.copyfile(PROJECT/name, self.project/name)
        for name in ('configs','patches'):
            shutil.copytree(PROJECT/name,self.project/name)
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
        with patch('run_all.subprocess.run') as command:
            command.return_value=argparse.Namespace(returncode=0,stderr='')
            job.cleanup()
        self.assertEqual([call.args[0] for call in command.call_args_list],
                         [['docker','stop','--time','15',cid],['docker','rm','--force',cid]])

    def test_model_partial_download_and_cache_reuse(self):
        destination=self.root/'model'
        destination.mkdir()
        (destination/'config.json').write_text('{}')
        revision='b'*40
        calls=[]
        def download(**kwargs):
            calls.append(kwargs)
            (destination/'tokenizer.json').write_text('{}')
            (destination/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'part.safetensors'}}))
            if len(calls)==1: raise OSError('interrupted download')
            (destination/'part.safetensors').write_bytes(b'fixture')
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError): prepare_model(destination,'org/model',revision,download)
            self.assertFalse((destination/'.hicache-model.json').exists())
            self.assertFalse(prepare_model(destination,'org/model',revision,download)['reused'])
            self.assertTrue(prepare_model(destination,'org/model',revision,download)['reused'])
            self.assertEqual(len(calls),2)
            (destination/'part.safetensors').unlink()
            self.assertFalse(prepare_model(destination,'org/model',revision,download)['reused'])
            self.assertTrue(calls[-1]['force_download'])


if __name__=='__main__': unittest.main()

"""命名配置的数据解析测试：不需要 SGLang、Docker 或 GPU。"""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import yaml

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serve_config", PROJECT / "serve_config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


class Configs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hicache-config-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.base = yaml.safe_load((PROJECT / "configs/serve/qwen3-8b-4k.yaml").read_text())
        self.write(self.base)

    def write(self, data):
        (self.directory / "test.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    def resolve(self, runtime="native", args=(), **env):
        return config.resolve("test", runtime, list(args), {"MODEL_ROOT": "/models with spaces", **env}, self.directory)

    def test_native_and_docker_equivalence_except_model_mount(self):
        native = self.resolve()
        docker = self.resolve("docker")
        self.assertEqual(native["model_directory"], "/models with spaces/Qwen3-8B")
        self.assertEqual(native["model_directory"], docker["model_directory"])
        self.assertEqual(docker["server"].pop("model-path"), "/model")
        self.assertEqual(native["server"].pop("model-path"), native["model_directory"])
        self.assertEqual(native["server"], docker["server"])
        self.assertEqual(native["environment"], docker["environment"])

    def test_cli_env_and_config_precedence(self):
        result = self.resolve(args=["--context-length", "3072", "--mem-fraction-static", "0.85"],
                              CONTEXT_LENGTH="2048", MEM_FRACTION="0.75")
        # YAML 记录环境覆盖，CLI 最后交给 SGLang，避免自行实现它的参数类型系统。
        self.assertEqual(result["server"]["context-length"], 2048)
        self.assertEqual(result["server"]["mem-fraction-static"], .75)
        self.assertEqual(result["cli_args"], ["--context-length", "3072", "--mem-fraction-static", "0.85"])
        self.assertEqual(result["server"]["max-total-tokens"], 8192)

    def test_model_override_handles_both_native_aliases(self):
        for args in (["--model-path", "/cli model"], ["--model=/cli model"]):
            result = self.resolve("docker", args, MODEL_DIR="/env model")
            self.assertEqual(result["model_directory"], "/cli model")
            self.assertEqual(result["server"]["model-path"], "/model")
            self.assertEqual(result["cli_args"], [])
        self.assertEqual(self.resolve(MODEL_DIR="/env model")["model_directory"], "/env model")
        for args in (["--model-path"], ["--model="], ["--model", ""]):
            with self.assertRaisesRegex(ValueError, "缺少路径"):
                self.resolve(args=args)

    def test_boolean_false_and_json_objects_survive_yaml(self):
        data = copy.deepcopy(self.base)
        data["sglang"]["enable-hierarchical-cache"] = False
        data["sglang"]["enable-hicache-experiment"] = False
        data["sglang"]["json-model-override-args"] = {"max_position_embeddings": 8192, "note": "quoted ' \" text"}
        data["env"]["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] = "0"
        self.write(data)
        result = self.resolve()
        restored = yaml.safe_load(result["server_yaml"])
        self.assertIs(restored["enable-hierarchical-cache"], False)
        self.assertEqual(restored["json-model-override-args"], data["sglang"]["json-model-override-args"])
        self.assertEqual(self.resolve(HICACHE="1")["server"]["enable-hierarchical-cache"], True)

    def test_hicache_disable_and_timer_validation(self):
        with self.assertRaisesRegex(ValueError, "GPU 计时"):
            self.resolve(HICACHE="0")
        result = self.resolve(HICACHE="0", FORWARD_TIMING="0")
        self.assertIs(result["server"]["enable-hierarchical-cache"], False)
        self.assertIs(result["server"]["enable-hicache-experiment"], False)
        self.assertEqual(result["environment"]["SGLANG_ENABLE_METRICS_DEVICE_TIMER"], "0")
        self.assertTrue(self.resolve(FORWARD_TIMING="0")["server"]["enable-hierarchical-cache"])

    def test_invalid_yaml_duplicate_keys_and_sections(self):
        for invalid in ("model: [", "model: A\nmodel: B\nsglang: {}", "[]", "model: ../outside\nsglang: {}",
                        "model: /absolute\nsglang: {}", "model: X\nsglang: []", "model: X\nsglang: {}\nother: 1",
                        "model: X\nsglang: {context_length: 4096}", "model: X\nsglang: {model: wrong}"):
            (self.directory / "test.yaml").write_text(invalid)
            with self.assertRaises((ValueError, yaml.YAMLError)):
                self.resolve()

    def test_config_cannot_be_stacked_or_execute_yaml_objects(self):
        for args in (["--config", "other.yaml"], ["--config=other.yaml"]):
            with self.assertRaisesRegex(ValueError, "不能再叠加"):
                self.resolve(args=args)
        (self.directory / "test.yaml").write_text("!!python/object/apply:os.system ['echo should-not-execute']")
        with self.assertRaises(yaml.YAMLError):
            self.resolve()


if __name__ == "__main__":
    unittest.main()

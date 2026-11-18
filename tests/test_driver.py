"""Driver 对服务、单次响应和离线汇总的约束。"""
import csv
from pathlib import Path
import tempfile
import unittest

from hicache.benchmark import check_server, validate
from hicache.prepare import write_json
from hicache.results import FIELDS, analyze, percentile


class DriverRegression(unittest.TestCase):
    def test_invalid_response_is_rejected(self):
        meta = dict(prompt_tokens=10, completion_tokens=1, cached_tokens=0, num_retractions=0,
                    first_token_latency=.001, e2e_latency=.002, queue_time=0,
                    finish_reason={"type": "length"})
        self.assertFalse(validate({"meta_info": meta}, 8, 2, 1, "cold")[0])
        for changes in ({"cached_tokens": 1}, {"cached_tokens_details": {"device": 0, "host": 1}},
                        {"num_retractions": 1}, {"prompt_tokens": 9}, {"completion_tokens": 2},
                        {"first_token_latency": None}, {"finish_reason": {"type": "abort"}}):
            with self.subTest(changes=changes):
                errors, _ = validate({"meta_info": meta | changes}, 8, 2, 1, "cold")
                self.assertTrue(errors)

    def test_incompatible_service_rejected_before_sampling(self):
        case = dict(id="p8_s2", prefix=range(8), suffix=range(2))
        info = dict(enable_metrics=True, disable_radix_cache=False, tp_size=1, pp_size=1,
                    dp_size=1, max_running_requests=1, context_length=64, page_size=1)
        check_server(info, [case], 1)
        for changes in ({"context_length": 10}, {"page_size": 32}, {"enable_hierarchical_cache": True},
                        {"max_total_num_tokens": 8}, {"tp_size": 3}, {"enable_metrics": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                check_server(info | changes, [case], 1)

    def test_invalid_latency_is_counted_but_excluded_from_percentiles(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4], .95), 3.85)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = dict(prefix_lengths=[8], suffix_lengths=[2], groups=["cold"], repeats=3)
            write_json(directory / "config.json", dict(status="complete", arguments=args))
            with (directory / "samples.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for valid, latency in [(True, 1), (True, 2), (True, 3), (False, 1_000_000)]:
                    writer.writerow(dict(prefix_length=8, suffix_length=2, group="cold",
                                         valid=valid, client_latency_ms=latency))
            analyze(directory, no_plot=True)
            with (directory / "summary.csv").open() as stream:
                row = next(r for r in csv.DictReader(stream) if r["metric"] == "client_latency_ms")
        self.assertEqual((row["valid"], row["invalid"]), ("3", "1"))
        self.assertEqual(float(row["median"]), 2)
        self.assertAlmostEqual(float(row["p95"]), 2.9)


if __name__ == "__main__":
    unittest.main()

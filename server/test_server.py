"""Protocol tests that do not require a GPU: python -m unittest discover -s server."""
import json
import subprocess
import threading
import unittest
from unittest.mock import patch
import server


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        server.RUNS.clear()

    def test_full_network_export_uses_native_generator(self):
        payload = {"totalNodes": 100000, "edgeCount": 2, "edges": [0, 1, 99998, 99999]}
        with patch.object(server.Path, "exists", return_value=True), patch.object(server.subprocess, "run") as run, patch.object(server, "automatic_execution_plan", return_value={"mpiRanks": 4}):
            run.return_value.stdout = json.dumps(payload)
            result = server.graph_preview(100000, 6, 42)
            self.assertEqual(result["totalNodes"], 100000)
            self.assertEqual(result["edges"], payload["edges"])
            self.assertEqual(result["executionPlan"]["mpiRanks"], 4)
            command = run.call_args.args[0]
            self.assertIn("--graph-only", command)
            self.assertIn("100000", command)

    def test_invalid_graph_never_starts_a_process(self):
        for args in [(0, 6, 42), (100, 100, 42), (100, 6, -1)]:
            with patch.object(server.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    server.graph_preview(*args)
                run.assert_not_called()

    def test_source_can_be_outside_old_preview(self):
        config = server.validate_config({"nodes": 100000, "sourceNode": 99999})
        self.assertEqual(config["sourceNode"], 99999)
        self.assertEqual(config["previewNodes"], 100000)

    def test_rank_choice_scales_with_graph_and_is_bounded(self):
        small = server.choose_execution_plan(10000, 6, 2, [15000])
        large = server.choose_execution_plan(100000, 6, 2, [15000])
        huge = server.choose_execution_plan(10000000, 6, 64, [15000])
        self.assertEqual(small["mpiRanks"], 2)
        self.assertEqual(large["mpiRanks"], 4)
        self.assertEqual(huge["mpiRanks"], 8)
        self.assertTrue(large["sharedGpu"])
        self.assertTrue(large["automatic"])
        self.assertEqual(server.choose_execution_plan(4, 2, 1, [15000])["mpiRanks"], 2)

    def test_rank_choice_respects_gpu_memory(self):
        low = server.choose_execution_plan(100000, 6, 8, [1000])
        self.assertEqual(low["mpiRanks"], 1)
        two_gpu = server.choose_execution_plan(100000, 6, 2, [1200, 1200])
        self.assertEqual(two_gpu["mpiRanks"], 4)

    def test_measured_choice_can_use_ten_ranks_on_the_matching_runtime(self):
        profile = {"nodes": 1000000, "meanDegree": 6, "gpuIds": ["GPU-test"], "mpiRanks": 10}
        self.assertEqual(server.measured_rank_choice(profile, 1000000, 6, ["GPU-test"], [15000]), 10)
        self.assertIsNone(server.measured_rank_choice(profile, 10000, 6, ["GPU-test"], [15000]))
        self.assertIsNone(server.measured_rank_choice(profile, 1000000, 8, ["GPU-test"], [15000]))
        self.assertIsNone(server.measured_rank_choice(profile, 1000000, 6, ["GPU-other"], [15000]))
        self.assertIsNone(server.measured_rank_choice(profile, 1000000, 6, ["GPU-test"], [1000]))
        for ranks in (0, 11, "10", True):
            self.assertIsNone(server.measured_rank_choice(dict(profile, mpiRanks=ranks), 1000000, 6, ["GPU-test"], [15000]))

    def test_old_rank_input_cannot_override_auto_selection(self):
        config = server.validate_config({"nodes": 100000, "sourceNode": 99999, "mpiRanks": 64})
        self.assertNotIn("mpiRanks", config)

    def test_colab_launch_allows_oversubscription_without_cpu_binding(self):
        config = server.validate_config({"nodes": 100000, "sourceNode": 99999})
        config["mpiRanks"] = 4
        with patch.object(server.os, "name", "posix"), patch.object(server.os, "geteuid", return_value=0, create=True):
            command = server.simulation_command(config)
        self.assertIn("--oversubscribe", command)
        self.assertEqual(command[command.index("--bind-to") + 1], "none")
        self.assertEqual(command[command.index("-np") + 1], "4")
        self.assertIn("--allow-run-as-root", command)

    def test_snapshots_bounded_without_mutating_inflight_events(self):
        server.RUNS["test"] = {"events": [], "done": False, "condition": threading.Condition(server.RUNS_LOCK)}
        first = {"kind": "progress", "tick": 0, "nodeStates": "0" * 99999 + "1"}
        server.append_event("test", first)
        for tick in range(1, 301):
            server.append_event("test", {"kind": "progress", "tick": tick, "nodeStates": "1" * 100000})
        events = server.RUNS["test"]["events"]
        self.assertEqual(len(events), 301)
        self.assertEqual(sum("nodeStates" in event for event in events), 1)
        self.assertEqual(len(events[-1]["nodeStates"]), 100000)
        self.assertIn("nodeStates", first)  # A stream already sending this event is safe.
        server.append_event("test", {"kind": "summary"})
        self.assertTrue(server.RUNS["test"]["done"])
        self.assertIn("nodeStates", events[-2])  # Late subscribers still get final colors.

    def test_native_errors_are_not_replaced_by_fake_graphs(self):
        with patch.object(server.Path, "exists", return_value=True), patch.object(server.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "engine", stderr="native failure")):
            with self.assertRaises(subprocess.CalledProcessError):
                server.graph_preview(100000, 6, 42)


if __name__ == "__main__":
    unittest.main()

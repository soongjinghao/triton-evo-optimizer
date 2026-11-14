import csv

import pytest

from config import EAConfig
from executor import TritonExecutor
from evolutionary_algorithm import EvolutionaryAlgorithm


def test_application_summary_uses_target_kernel_median(tmp_path):
    summary = tmp_path / "PROF_1" / "mindstudio_profiler_output" / "op_summary_1.csv"
    summary.parent.mkdir(parents=True)
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Op Name", "Task Duration(us)"])
        writer.writeheader()
        writer.writerows([
            {"Op Name": "_target", "Task Duration(us)": "5.0"},
            {"Op Name": "_other", "Task Duration(us)": "900.0"},
            {"Op Name": "_target", "Task Duration(us)": "7.0"},
            {"Op Name": "_target", "Task Duration(us)": "NA"},
            {"Op Name": "_target", "Task Duration(us)": "6.0"},
        ])

    assert TritonExecutor._parse_application_summary(
        tmp_path, "_target", min_samples=3
    ) == 6.0
    assert TritonExecutor._parse_application_summary(
        tmp_path, "_target", min_samples=4
    ) is None


def test_application_backend_remeasures_and_caches_seed_baseline(tmp_path, monkeypatch):
    (tmp_path / "kernel.py").write_text("# seed source\n", encoding="utf-8")
    test_file = tmp_path / "test_kernel.py"
    test_file.write_text("print('ok')\n", encoding="utf-8")

    config = EAConfig()
    config.msprof_backend = "application"
    executor = TritonExecutor(
        baseline_time=5.0,
        test_code_path=str(test_file),
        config=config,
        kernel_name="kernel",
        work_dir=tmp_path,
    )
    calls = []

    def profile(*args, **kwargs):
        calls.append((args, kwargs))
        return 7.25

    monkeypatch.setattr(executor, "_run_msprof_application", profile)

    assert executor._baseline_for_device(0) == 7.25
    assert executor._baseline_for_device(0) == 7.25
    assert executor.baseline_time == 7.25
    assert len(calls) == 1


def test_application_backend_rejects_formal_manifest_baselines(tmp_path):
    test_file = tmp_path / "test_kernel.py"
    test_file.write_text("print('ok')\n", encoding="utf-8")
    config = EAConfig()
    config.msprof_backend = "application"
    config.require_manifest_baseline = True

    with pytest.raises(ValueError, match="incompatible with formal manifest-baseline"):
        TritonExecutor(
            baseline_time=5.0,
            test_code_path=str(test_file),
            config=config,
            kernel_name="kernel",
            work_dir=tmp_path,
        )


def test_application_profile_is_injected_into_strategy_context(tmp_path):
    class ProfileExecutor:
        performance_dir = tmp_path
        kernel_name = "kernel"
        baseline_time = 8.0
        msprof_backend = "application"

        @staticmethod
        def latest_application_summary_rows(device_id):
            return [
                {
                    "Task Duration(us)": "2.0",
                    "aiv_vec_ratio": "0.25",
                    "aiv_scalar_ratio": "0.50",
                },
                {
                    "Task Duration(us)": "4.0",
                    "aiv_vec_ratio": "0.75",
                    "aiv_scalar_ratio": "0.50",
                },
            ]

    algorithm = EvolutionaryAlgorithm.__new__(EvolutionaryAlgorithm)
    algorithm.executor = ProfileExecutor()
    algorithm.seed_execution_time = 3.0

    context = algorithm._extract_profiling_context()

    assert "应用级种子基线=8.00 us" in context
    assert "官方基线" not in context
    assert "目标算子记录=2, Task Duration 中位数=3.000 us" in context
    assert "Vector=50.0%, Scalar=50.0%" in context

"""Run audit, fixed sampling, both training variants, final tests and comparison."""
from pathlib import Path

from common import load_config, parser, project_path
from check_dataset import audit
from compare import compare
from common import write_csv, write_json
from evaluate import evaluate
from prepare_train_subset import prepare
from train import train


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--improved-config", default="configs/improved.yaml")
    cli.add_argument("--report-dir", default="outputs")
    cli.add_argument("--run-dir", help="Place both new experiments and comparison under this new directory")
    args = cli.parse_args()
    baseline = load_config(args)
    args.config = args.improved_config
    improved = load_config(args)
    if args.run_dir:
        run_dir = project_path(args.run_dir)
        baseline["output_dir"] = str(run_dir / "baseline")
        improved["output_dir"] = str(run_dir / "improved")
        baseline["audit_dir"] = improved["audit_dir"] = str(run_dir / "audit")
        args.report_dir = str(run_dir)
    # Fail before expensive work if settings would make the planned comparison unfair.
    for key in ("data_root", "subset_path", "model", "pretrained", "pretrained_weights", "image_size", "train_samples_per_class",
                "random_seed", "batch_size", "epochs", "learning_rate", "device", "num_workers", "torch_threads"):
        if baseline[key] != improved[key]:
            raise ValueError(f"Both configs must agree on {key}")
    if baseline["freeze_backbone"] is not True or improved["freeze_backbone"] is not False:
        raise ValueError("Expected baseline freeze_backbone=true and improved=false")
    if baseline["output_dir"] == improved["output_dir"]:
        raise ValueError("Use separate output directories for the two experiments")
    for cfg in (baseline, improved):
        out = Path(cfg["output_dir"])
        if any((out / name).exists() for name in ("best.pt", "history.csv", "training_summary.json")):
            raise FileExistsError(f"Existing experiment: {out}. Set new output_dir values in the configs.")
    audit(baseline)
    prepare(baseline)
    train(baseline)
    train(improved)
    results = [evaluate(cfg, Path(cfg["output_dir"]) / "best.pt", Path(cfg["output_dir"])) for cfg in (baseline, improved)]
    comparison = compare(*results)
    report = project_path(args.report_dir)
    write_json(report / "comparison.json", comparison)
    write_csv(report / "comparison.csv", comparison["experiments"], list(comparison["experiments"][0]))
    print(f"Pipeline complete: {report / 'comparison.csv'}", flush=True)

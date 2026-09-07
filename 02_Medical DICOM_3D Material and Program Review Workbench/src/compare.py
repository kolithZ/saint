"""Compare predeclared experiments only when data and key settings match."""
import argparse
import json

from common import project_path, write_csv, write_json


def compare(baseline: dict, improved: dict) -> dict:
    for key in ("manifest_sha256", "train_data_sha256", "val_data_sha256", "test_data_sha256", "test_found", "test_used"):
        if baseline[key] != improved[key]:
            raise ValueError(f"Experiments are not comparable: different {key}")
    if baseline["pretrained_weights"]["sha256"] != improved["pretrained_weights"]["sha256"]:
        raise ValueError("Experiments use different pretrained weights")
    for key in ("model", "pretrained", "random_seed", "image_size", "batch_size", "epochs", "learning_rate", "device", "num_workers", "torch_threads"):
        if baseline["training_config"][key] != improved["training_config"][key]:
            raise ValueError(f"Experiments are not comparable: different {key}")
    if baseline["training_config"]["freeze_backbone"] is not True or improved["training_config"]["freeze_backbone"] is not False:
        raise ValueError("Expected frozen baseline and layer4-finetuned improved model")
    rows = [{"experiment": name, **{key: metrics[key] for key in ("best_epoch", "n", "sensitivity", "roc_auc", "specificity", "accuracy", "tp", "tn", "fp", "fn")}}
            for name, metrics in (("baseline", baseline), ("layer4_finetune", improved))]
    return {"experiments": rows, "improved_minus_baseline": {
        key: improved[key] - baseline[key] if improved[key] is not None and baseline[key] is not None else None
        for key in ("sensitivity", "roc_auc", "specificity", "accuracy")},
        "interpretation": "One fixed-seed, small-sample comparison; a change in point estimates does not establish statistical or clinical superiority."}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", default="outputs/baseline/metrics.json")
    cli.add_argument("--improved", default="outputs/improved/metrics.json")
    cli.add_argument("--output-dir", default="outputs")
    args = cli.parse_args()
    result = compare(json.loads(project_path(args.baseline).read_text()), json.loads(project_path(args.improved).read_text()))
    output = project_path(args.output_dir)
    write_json(output / "comparison.json", result)
    write_csv(output / "comparison.csv", result["experiments"], list(result["experiments"][0]))
    print(json.dumps(result, ensure_ascii=False, indent=2))

#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
import sys
from typing import Optional, Tuple, List
import copy
from datetime import datetime


def resolve_repo_paths() -> Tuple[str, str, str, str]:
    """Return absolute paths for e2e_root (this dir), parent_root, nnunet_root, ai_ene_root."""
    e2e_root = os.path.dirname(os.path.abspath(__file__))  # .../AI_ENE
    parent_root = os.path.abspath(os.path.join(e2e_root, os.pardir))  # repo parent (pene_survival)
    nnunet_root = os.path.join(e2e_root, "nnunet_segmentation")
    ai_ene_root = os.path.join(e2e_root, "ENE_inference")
    return e2e_root, parent_root, nnunet_root, ai_ene_root


def read_yaml_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def write_yaml_config(config_path: str, cfg: dict) -> None:
    import yaml
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)


def ensure_abs_path(path_value: str, base_dir: str) -> str:
    expanded = os.path.expanduser(path_value)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(base_dir, expanded))


def run_subprocess(cmd: List[str], cwd: Optional[str] = None) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def discover_nnUNet_outputs(nnunet_root: str, task_name: str) -> Tuple[str, str]:
    task_dir = os.path.join(
        nnunet_root, "nnUNet_output", "nnUNet_raw_data", task_name
    )
    imagesTs = os.path.join(task_dir, "imagesTs")
    predsTs = os.path.join(task_dir, "predsTs")
    if not os.path.isdir(imagesTs):
        raise FileNotFoundError(f"imagesTs not found: {imagesTs}")
    if not os.path.isdir(predsTs):
        raise FileNotFoundError(f"predsTs not found: {predsTs}")
    return imagesTs, predsTs


def find_latest_csv(output_dir: str) -> Optional[str]:
    pattern = os.path.join(output_dir, "prediction_list", "predictions_*", "*.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main() -> None:
    e2e_root, parent_root, nnunet_root, ai_ene_root = resolve_repo_paths()

    default_dicom_dir = os.path.join(nnunet_root, "dicom_data")
    # Default outputs under project: AI_ENE/output
    default_output_dir = os.path.join(e2e_root, "output")

    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: DICOM -> nnUNet segmentation -> AI-ENE classification"
    )
    # Unified input directory flag (preferred)
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Unified input directory for either DICOM or non-DICOM inputs. "
             "Interpretation depends on --input-format (or config). "
             "If format is 'dicom', this is the DICOM root; otherwise it should contain NRRD/NIfTI.",
    )
    parser.add_argument(
        "--dicom-dir",
        default=default_dicom_dir,
        help="Path to input DICOM directory (default: nnunet_segmentation/dicom_data)",
    )
    parser.add_argument(
        "--input-data-dir",
        default=None,
        help="Path to non-DICOM input directory (NRRD/NIfTI). Overrides dicom when --input-format != dicom",
    )
    parser.add_argument(
        "--input-format",
        default=None,
        choices=["dicom", "nrrd", "nii", "nii.gz"],
        help="Input data format. When omitted, config.yaml governs (default: dicom)",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Path to base output directory for AI-ENE results (default: AI_ENE/output)",
    )
    parser.add_argument(
        "--gpu",
        default="cpu",
        help="GPU index (e.g., 0), 'cpu', or 'auto' (default: cpu)",
    )
    parser.add_argument(
        "--include-short-axes",
        action="store_true",
        help="Include short/long axis metrics in CSV output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of cases to process in AI-ENE (default: all)",
    )
    parser.add_argument(
        "--fg-labels",
        default="2",
        help="Foreground label(s) in segmentation, comma-separated, or 'any' (default: 2)",
    )
    args = parser.parse_args()

    # Global device visibility for both nnUNet (PyTorch) and classification (TensorFlow)
    try:
        gpu_arg = str(args.gpu).lower() if args.gpu is not None else "auto"
        if gpu_arg == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("CUDA_VISIBLE_DEVICES cleared (CPU mode).")
        elif gpu_arg == "auto":
            # Do not restrict; visibility decided by container/runtime
            print("CUDA visibility: auto (respecting container runtime).")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(f"CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")
    except Exception:
        # Do not fail E2E if env setting fails
        pass

    # Enforce explicit CLI usage to avoid ambiguity and hidden defaults
    required_flags = ["--input-dir", "--input-format", "--output-dir", "--gpu"]
    missing = [f for f in required_flags if f not in sys.argv]
    if missing or not args.input_dir or not args.input_format:
        msg_lines = [
            "Error: Missing required CLI options.",
            "Mandatory flags: --input-dir PATH  --input-format {dicom|nrrd|nii|nii.gz}  --output-dir PATH  --gpu {cpu|auto|0}",
            "Example (DICOM):",
            "  python run_e2e.py \\",
            "    --input-dir /abs/path/to/dicom_root \\",
            "    --input-format dicom \\",
            "    --output-dir output \\",
            "    --gpu cpu",
        ]
        if missing:
            msg_lines.append(f"Missing flags: {' '.join(missing)}")
        print("\n".join(msg_lines), file=sys.stderr)
        sys.exit(2)

    # 1) Update nnUNet config with DICOM directory (temporarily; restore after run)
    config_path = os.path.join(nnunet_root, "src", "config.yaml")
    cfg = read_yaml_config(config_path)
    original_cfg = copy.deepcopy(cfg)
    cfg_path_section = cfg.setdefault("path", {})
    cfg_data_section = cfg.setdefault("data", {})

    # Apply CLI overrides (temporary)
    # Resolve effective format: CLI > config > default 'dicom'
    if args.input_format:
        cfg_data_section["input_data_format"] = args.input_format
    input_format_effective = cfg_data_section.get("input_data_format", "dicom")

    # Back-compat handling and precedence:
    # If --input-dir is provided, it takes precedence over --dicom-dir/--input-data-dir.
    # If not, fall back to legacy flags.
    if args.input_dir:
        unified_path = ensure_abs_path(args.input_dir, base_dir=nnunet_root)
        if input_format_effective == "dicom":
            cfg_path_section["dicom_dir"] = unified_path
            print(f"Updated nnUNet config dicom_dir (from --input-dir): {cfg_path_section['dicom_dir']}")
        else:
            cfg_path_section["input_data_dir"] = unified_path
            print(f"Updated nnUNet config input_data_dir (from --input-dir): {cfg_path_section['input_data_dir']}")
        if args.dicom_dir or args.input_data_dir:
            print("Note: --input-dir provided; ignoring legacy --dicom-dir/--input-data-dir for this run.")
    else:
        # Legacy flags
        cfg_path_section["dicom_dir"] = ensure_abs_path(args.dicom_dir, base_dir=nnunet_root)
        if args.input_data_dir:
            cfg_path_section["input_data_dir"] = ensure_abs_path(args.input_data_dir, base_dir=nnunet_root)
        print(f"Updated nnUNet config dicom_dir: {cfg_path_section['dicom_dir']}")
        if args.input_data_dir:
            print(f"Updated nnUNet config input_data_dir: {cfg_path_section['input_data_dir']}")
        if args.input_format:
            print(f"Updated nnUNet config input_data_format: {cfg_data_section['input_data_format']}")

    # Determine task name early for asset checks
    task_name = cfg.get("inference", {}).get("task_name", "Task501_ENE")

    # Startup checks for required assets (template and trained model)
    explicit_template = cfg_path_section.get("registration_template_path")
    repo_template_path = os.path.join(nnunet_root, "reg_img_template", "template_image.nrrd")
    template_ok = False
    if explicit_template and os.path.isfile(explicit_template):
        template_ok = True
    elif os.path.isfile(repo_template_path):
        template_ok = True
    if not template_ok:
        print(
            "Error: Registration template not found.\n"
            f"- Expected at config path 'path.registration_template_path' or at repository default:\n"
            f"  {repo_template_path}\n"
            "See README 'Downloads' section for asset placement.",
            file=sys.stderr,
        )
        sys.exit(2)

    model_base = os.path.join(nnunet_root, "nnUNet_trained_model", "nnUNet", task_name)
    model_plan = os.path.join(model_base, "plans.pkl")
    fold_dir = os.path.join(model_base, "fold_0")
    model_ckpt = os.path.join(fold_dir, "model_final_checkpoint.model")
    model_pkl = os.path.join(fold_dir, "model_final_checkpoint.model.pkl")
    missing_model_files = [p for p in [model_plan, model_ckpt, model_pkl] if not os.path.isfile(p)]
    if missing_model_files:
        msg = ["Error: Missing nnUNet trained model files:"]
        msg.extend([f"- {p}" for p in missing_model_files])
        msg.append("Place them under 'AI_ENE/nnunet_segmentation/nnUNet_trained_model/nnUNet/<Task>/...' per README Downloads.")
        print("\n".join(msg), file=sys.stderr)
        sys.exit(2)

    # Always use a unique, isolated workspace per run under nnunet_segmentation/runs/
    runs_dir = os.path.join(nnunet_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    workspace_root = os.path.join(runs_dir, run_id)
    os.makedirs(workspace_root, exist_ok=True)
    cfg_path_section["project_dir"] = workspace_root
    print(f"Using isolated project_dir: {workspace_root}")

    # Ensure trained model is available in the workspace by linking/copying from the repository model location
    src_trained_model_dir = os.path.join(nnunet_root, "nnUNet_trained_model")
    dst_trained_model_dir = os.path.join(workspace_root, "nnUNet_trained_model")
    if os.path.isdir(src_trained_model_dir):
        try:
            if not os.path.exists(dst_trained_model_dir):
                os.symlink(src_trained_model_dir, dst_trained_model_dir)
                print(f"Linked trained model into workspace: {dst_trained_model_dir} -> {src_trained_model_dir}")
        except Exception:
            # Fallback to shallow copy (directory structure and files)
            import shutil as _shutil
            if not os.path.exists(dst_trained_model_dir):
                _shutil.copytree(src_trained_model_dir, dst_trained_model_dir)
                print(f"Copied trained model into workspace: {dst_trained_model_dir}")
    else:
        print(f"Warning: Trained model directory not found: {src_trained_model_dir}. "
              f"Ensure models are available under workspace nnUNet_trained_model.")

    # Prefer using a stable registration template in repo if not explicitly set
    if not cfg_path_section.get("registration_template_path"):
        repo_template_path = os.path.join(nnunet_root, "reg_img_template", "template_image.nrrd")
        if os.path.isfile(repo_template_path):
            cfg_path_section["registration_template_path"] = repo_template_path
            print(f"Using registration template from repo: {repo_template_path}")
        else:
            # Leave unset; preprocessing will expect it under workspace main_data/reg_img_template/template_image.nrrd
            print("No registration_template_path set; expected repo path AI_ENE/nnunet_segmentation/reg_img_template/template_image.nrrd or workspace main_data/reg_img_template/template_image.nrrd")

    write_yaml_config(config_path, cfg)

    # Read task name from config for downstream paths
    task_name = cfg.get("inference", {}).get("task_name", "Task501_ENE")

    try:
        # 2) Run nnUNet pipeline
        run_subprocess(
            [sys.executable, os.path.join(nnunet_root, "src", "run_pipeline.py")],
            cwd=nnunet_root,
        )

        # 3) Discover nnUNet outputs (from the active workspace/project_dir)
        base_for_outputs = cfg.get("path", {}).get("project_dir") or nnunet_root
        imagesTs_dir, predsTs_dir = discover_nnUNet_outputs(base_for_outputs, task_name)
        print(f"nnUNet imagesTs: {imagesTs_dir}")
        print(f"nnUNet predsTs:  {predsTs_dir}")

        # 4) Run AI-ENE classification
        out_base = ensure_abs_path(args.output_dir, base_dir=parent_root)
        classification_cmd: List[str] = [
            sys.executable,
            os.path.join(ai_ene_root, "ene_classification.py"),
            "--seg-dir",
            predsTs_dir,
            "--img-dir",
            imagesTs_dir,
            "--output-dir",
            out_base,
            "--gpu",
            str(args.gpu),
            "--verbose",
        ]
        if args.include_short_axes:
            classification_cmd.append("--include-short-axes")
        if args.limit is not None:
            classification_cmd.extend(["--limit", str(args.limit)])
        if args.fg_labels:
            classification_cmd.extend(["--fg-labels", str(args.fg_labels)])

        run_subprocess(classification_cmd, cwd=ai_ene_root)

        # 5) Locate and print path to latest CSV
        latest_csv = find_latest_csv(out_base)
        if latest_csv:
            print(f"Done. Check outputs under: {os.path.dirname(os.path.dirname(latest_csv))}")
            print(f"CSV: {latest_csv}")
        else:
            print(f"Done. Check outputs under: {out_base}")
            print("Warning: No CSV found. Verify prediction_list contents.")
    finally:
        # Restore original config to avoid persisting CLI overrides
        write_yaml_config(config_path, original_cfg)
        print("Restored config.yaml to original state")


if __name__ == "__main__":
    main()



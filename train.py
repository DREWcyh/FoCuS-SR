import os
import gc
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from model import CSDLoss, FoCuSSR
from src.my_utils.training_utils import parse_args  
from src.datasets.dataset import PairedSROnlineTxtDataset

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import random
import yaml
import json
from collections import defaultdict

PIX_ADAPTERS = ['default_encoder_pix', 'default_decoder_pix', 'default_others_pix']
PIX_SEM_ADAPTERS = [
    'default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
    'default_encoder_sem', 'default_decoder_sem', 'default_others_sem',
]
LAYER_DIAGNOSTIC_GRAD_SCALES = {
    "l2": 65536.0,
    "lpips": 1.0,
    "csd": 1.0,
}


def unwrap_focussr_model(model):
    return model.module if hasattr(model, "module") else model


def set_active_adapters(model, adapter_names):
    unwrap_focussr_model(model).unet.set_adapter(adapter_names)


def semantic_adapter_names_for_training(model, args):
    focussr = unwrap_focussr_model(model)
    if getattr(focussr, "semantic_adapter_mode", "legacy") == "sequential":
        phase = args.semantic_train_phase if args.semantic_train_phase != "legacy" else "both"
        return focussr.pixel_semantic_adapter_names_for_phase(phase)
    return PIX_SEM_ADAPTERS


def activate_semantic_training(model, args):
    focussr = unwrap_focussr_model(model)
    adapter_names = semantic_adapter_names_for_training(model, args)
    focussr.unet.set_adapter(adapter_names)
    if getattr(focussr, "semantic_adapter_mode", "legacy") == "sequential":
        phase = args.semantic_train_phase if args.semantic_train_phase != "legacy" else "both"
        focussr.set_train_semantic_roles(phase)
    else:
        focussr.set_train_sem()
    return adapter_names


def lora_module_from_param_name(name):
    if ".lora_" in name:
        return name.split(".lora_", 1)[0]
    return None


def build_lora_optimizer_params(net_focussr, args):
    layers_to_opt = []
    include_frozen_for_late_semantic_stage = int(getattr(args, "pix_steps", 0)) > 0

    for name, param in net_focussr.unet.named_parameters():
        if "lora" not in name:
            continue
        if not include_frozen_for_late_semantic_stage and not param.requires_grad:
            continue
        layers_to_opt.append(param)

    metadata = {
        "num_trainable_lora_param_tensors": len(layers_to_opt),
        "optimizer_includes_frozen_for_late_semantic_stage": include_frozen_for_late_semantic_stage,
    }
    return layers_to_opt, layers_to_opt, metadata


def save_hparams(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "hparams.yml"), "w") as f:
        yaml.safe_dump(dict(vars(args)), f, sort_keys=True)


def sanitize_tracker_config(args):
    tracker_config = {}
    for key, value in vars(args).items():
        if isinstance(value, (int, float, str, bool, torch.Tensor)):
            tracker_config[key] = value
        else:
            tracker_config[key] = str(value)
    return tracker_config


def load_rank_curriculum_metadata(args):
    pixel_path = getattr(args, "pixel_rank_curriculum_json", None)
    path = getattr(args, "semantic_rank_curriculum_json", None)
    sequential_path = getattr(args, "sequential_semantic_lora_curriculum_json", None)
    pixel_curriculum = None
    if not pixel_path:
        args.pixel_rank_curriculum = None
        args.pixel_rank_curriculum_num_modules = 0
        args.pixel_rank_curriculum_budget_relative_error = 0.0
    else:
        with open(pixel_path, "r", encoding="utf-8") as f:
            pixel_curriculum = json.load(f)
        args.pixel_rank_curriculum = pixel_curriculum
        args.pixel_rank_curriculum_num_modules = len(pixel_curriculum.get("rank_pattern", {}))
        args.pixel_rank_curriculum_budget_relative_error = float(pixel_curriculum.get("budget_relative_error", 0.0))

    curriculum = None
    if not path:
        args.semantic_rank_curriculum = None
    else:
        with open(path, "r", encoding="utf-8") as f:
            curriculum = json.load(f)
        args.semantic_rank_curriculum = curriculum
        args.semantic_rank_curriculum_num_modules = len(curriculum.get("rank_pattern", {}))
        args.semantic_rank_curriculum_budget_relative_error = float(curriculum.get("budget_relative_error", 0.0))

    if not path:
        args.semantic_rank_curriculum_num_modules = 0
        args.semantic_rank_curriculum_budget_relative_error = 0.0

    if not sequential_path:
        args.sequential_semantic_lora_curriculum = None
        args.stage2a_low_conflict_module_count = 0
        args.stage2b_high_conflict_module_count = 0
    else:
        with open(sequential_path, "r", encoding="utf-8") as f:
            sequential_curriculum = json.load(f)
        args.sequential_semantic_lora_curriculum = sequential_curriculum
        args.stage2a_low_conflict_module_count = len(sequential_curriculum.get("stage2a_low_conflict_modules", []) or [])
        args.stage2b_high_conflict_module_count = len(sequential_curriculum.get("stage2b_high_conflict_modules", []) or [])
        args.sequential_semantic_conflict_top_ratio = float(sequential_curriculum.get("conflict_top_ratio", 0.0))
        args.sequential_semantic_score_definition = sequential_curriculum.get("score_definition", {})


def parse_layer_diagnostic_objectives(value):
    allowed = {"l2", "lpips", "csd"}
    objectives = []
    for item in str(value).split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported layer diagnostic objective: {item}")
        if item not in objectives:
            objectives.append(item)
    if not objectives:
        raise ValueError("layer_diagnostic_objectives cannot be empty")
    return objectives


def should_log_layer_diagnostics(args, global_step):
    if not getattr(args, "enable_layer_diagnostic_logging", False):
        return False
    freq = int(getattr(args, "layer_diagnostic_freq", 0))
    if freq <= 0:
        return False
    start_step = int(getattr(args, "layer_diagnostic_start_step", 1))
    if global_step < start_step:
        return False
    return (global_step - start_step) % freq == 0


def current_lora_train_stage(net_focussr):
    model = unwrap_focussr_model(net_focussr)
    pixel_count = 0
    semantic_count = 0
    for name, param in model.unet.named_parameters():
        if "lora" not in name or not param.requires_grad:
            continue
        if "sem" in name:
            semantic_count += 1
        elif "pix" in name:
            pixel_count += 1
    if semantic_count > 0:
        return "semantic", semantic_count
    if pixel_count > 0:
        return "pixel", pixel_count
    return "unknown", 0


def collect_lora_diagnostic_params(net_focussr, stage):
    model = unwrap_focussr_model(net_focussr)
    pattern = "sem" if stage == "semantic" else "pix"
    selected = []
    module_to_param_names = defaultdict(list)
    for name, param in model.unet.named_parameters():
        if "lora" not in name or pattern not in name or not param.requires_grad:
            continue
        module = lora_module_from_param_name(name)
        if module is None:
            continue
        selected.append((name, param, module))
        module_to_param_names[module].append(name)
    return selected, module_to_param_names


def layer_diagnostic_forward_loss(net_focussr, net_lpips, net_csd, batch, args, objective):
    x_src = batch["conditioning_pixel_values"]
    x_tgt = batch["output_pixel_values"]
    y_pred, latents_pred, prompt_embeds, neg_prompt_embeds = net_focussr(
        x_src,
        x_tgt,
        batch=batch,
        args=args,
        force_null_text=False,
        deterministic_vae=True,
    )
    if objective == "l2":
        return F.mse_loss(y_pred.float(), x_tgt.float(), reduction="mean")
    if objective == "lpips":
        return net_lpips(y_pred.float(), x_tgt.float()).mean()
    if objective == "csd":
        return net_csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args)
    raise ValueError(f"Unsupported layer diagnostic objective: {objective}")


def compute_layer_diagnostic_grads(net_focussr, net_lpips, net_csd, batch, args, selected, objectives):
    names = [item[0] for item in selected]
    params = [item[1] for item in selected]
    losses = {}
    grad_maps = {}
    for objective in objectives:
        loss = layer_diagnostic_forward_loss(net_focussr, net_lpips, net_csd, batch, args, objective)
        grad_scale = float(LAYER_DIAGNOSTIC_GRAD_SCALES.get(objective, 1.0))
        scaled_loss = loss.float() * grad_scale
        grads = torch.autograd.grad(
            scaled_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        losses[objective] = float(loss.detach().cpu().item())
        grad_maps[objective] = {
            name: None if grad is None else grad.detach().float().cpu() / grad_scale
            for name, grad in zip(names, grads)
        }
        del loss, scaled_loss, grads
    return losses, grad_maps


def tensor_dot_value(a, b):
    if a is None or b is None:
        return 0.0
    return float(torch.dot(a.reshape(-1), b.reshape(-1)).item())


def tensor_norm_sq_value(tensor):
    if tensor is None:
        return 0.0
    flat = tensor.reshape(-1)
    return float(torch.dot(flat, flat).item())


def rank_values(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and values[order[end]] == values[order[idx]]:
            end += 1
        rank = 0.5 * (idx + end - 1) + 1.0
        ranks[order[idx:end]] = rank
        idx = end
    return ranks


def spearman_correlation(xs, ys):
    if len(xs) < 2:
        return None
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if float(np.std(xs)) < 1e-12 or float(np.std(ys)) < 1e-12:
        return None
    rx = rank_values(xs)
    ry = rank_values(ys)
    corr = np.corrcoef(rx, ry)[0, 1]
    if not np.isfinite(corr):
        return None
    return float(corr)


def safe_mean(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return None
    return float(np.mean(values))


def top_layer_preview(layers, key, n=10, reverse=True):
    rows = sorted(layers, key=lambda item: item.get(key, 0.0), reverse=reverse)[:n]
    return [
        {
            "module": row["module"],
            key: row.get(key, 0.0),
            "importance_l2": row.get("importance_l2", 0.0),
            "importance_semantic": row.get("importance_semantic", 0.0),
            "conflict_raw": row.get("conflict_raw", 0.0),
        }
        for row in rows
    ]


def build_layer_diagnostic_rows(selected, module_to_param_names, grad_maps, objectives):
    norm_sq_by_module = defaultdict(dict)
    for module, param_names in module_to_param_names.items():
        for objective in objectives:
            norm_sq_by_module[module][objective] = sum(
                tensor_norm_sq_value(grad_maps[objective].get(param_name))
                for param_name in param_names
            )

    totals = {}
    for objective in objectives:
        totals[objective] = sum(
            float(np.sqrt(max(norm_sq_by_module[module].get(objective, 0.0), 0.0)))
            for module in module_to_param_names
        )

    rows = []
    for module in sorted(module_to_param_names):
        row = {
            "module": module,
            "param_tensors": len(module_to_param_names[module]),
            "param_count": int(sum(
                param.numel() for name, param, item_module in selected if item_module == module
            )),
        }
        for objective in ("l2", "lpips", "csd"):
            norm_sq = norm_sq_by_module[module].get(objective, 0.0)
            grad_norm = float(np.sqrt(max(norm_sq, 0.0)))
            total = totals.get(objective, 0.0)
            row[f"grad_norm_{objective}"] = grad_norm
            row[f"importance_{objective}"] = grad_norm / (total + 1e-12) if objective in objectives else 0.0

        if "l2" in objectives and "csd" in objectives:
            dot = sum(
                tensor_dot_value(grad_maps["l2"].get(param_name), grad_maps["csd"].get(param_name))
                for param_name in module_to_param_names[module]
            )
            norm_l2 = row["grad_norm_l2"]
            norm_csd = row["grad_norm_csd"]
            cos_l2_csd = dot / (norm_l2 * norm_csd + 1e-12)
            cos_l2_csd = float(max(min(cos_l2_csd, 1.0), -1.0))
        else:
            dot = 0.0
            cos_l2_csd = 0.0

        row["dot_l2_csd"] = float(dot)
        row["cos_l2_csd"] = cos_l2_csd
        row["conflict_raw"] = -cos_l2_csd
        row["conflict_relu"] = max(0.0, -cos_l2_csd)
        row["importance_semantic"] = row["importance_lpips"] + row["importance_csd"]
        rows.append(row)

    for key, rank_key, reverse in (
        ("importance_l2", "rank_importance_l2", True),
        ("importance_lpips", "rank_importance_lpips", True),
        ("importance_csd", "rank_importance_csd", True),
        ("importance_semantic", "rank_importance_semantic", True),
        ("conflict_raw", "rank_conflict_raw", True),
        ("conflict_relu", "rank_conflict_relu", True),
    ):
        ordered = sorted(rows, key=lambda item: item.get(key, 0.0), reverse=reverse)
        for rank, row in enumerate(ordered, start=1):
            row[rank_key] = rank

    rows.sort(key=lambda item: item.get("rank_conflict_raw", len(rows) + 1))
    return rows, totals


def build_layer_diagnostic_summary(step, stage, objectives, losses, layers, totals, selected):
    conflict_relu = [row.get("conflict_relu", 0.0) for row in layers]
    conflict_raw = [row.get("conflict_raw", 0.0) for row in layers]
    cos_l2_csd = [row.get("cos_l2_csd", 0.0) for row in layers]
    importance_l2 = [row.get("importance_l2", 0.0) for row in layers]
    importance_semantic = [row.get("importance_semantic", 0.0) for row in layers]
    global_dot_l2_csd = sum(float(row.get("dot_l2_csd", 0.0)) for row in layers)
    global_norm_l2 = float(np.sqrt(sum(float(row.get("grad_norm_l2", 0.0)) ** 2 for row in layers)))
    global_norm_csd = float(np.sqrt(sum(float(row.get("grad_norm_csd", 0.0)) ** 2 for row in layers)))
    global_cos_l2_csd = global_dot_l2_csd / (global_norm_l2 * global_norm_csd + 1e-12)
    global_cos_l2_csd = float(max(min(global_cos_l2_csd, 1.0), -1.0))
    summary = {
        "step": int(step),
        "stage": stage,
        "objectives": list(objectives),
        "diagnostic_grad_scales": {
            objective: float(LAYER_DIAGNOSTIC_GRAD_SCALES.get(objective, 1.0))
            for objective in objectives
        },
        "losses": losses,
        "num_layers": len(layers),
        "selected_param_tensors": len(selected),
        "selected_param_count": int(sum(param.numel() for _name, param, _module in selected)),
        "importance_totals": totals,
        "mean_cos_l2_csd": safe_mean(cos_l2_csd),
        "global_cos_l2_csd": global_cos_l2_csd,
        "global_conflict_raw": -global_cos_l2_csd,
        "global_conflict_relu": max(0.0, -global_cos_l2_csd),
        "global_dot_l2_csd": global_dot_l2_csd,
        "min_cos_l2_csd": float(min(cos_l2_csd)) if cos_l2_csd else None,
        "max_cos_l2_csd": float(max(cos_l2_csd)) if cos_l2_csd else None,
        "negative_cos_l2_csd_ratio": (
            float(np.mean([value < 0.0 for value in cos_l2_csd])) if cos_l2_csd else None
        ),
        "mean_conflict_relu": safe_mean(conflict_relu),
        "mean_conflict_raw": safe_mean(conflict_raw),
        "spearman_importance_l2_vs_conflict_relu": spearman_correlation(importance_l2, conflict_relu),
        "spearman_importance_l2_vs_conflict_raw": spearman_correlation(importance_l2, conflict_raw),
        "spearman_importance_semantic_vs_conflict_relu": spearman_correlation(importance_semantic, conflict_relu),
        "spearman_importance_semantic_vs_conflict_raw": spearman_correlation(importance_semantic, conflict_raw),
        "top_importance_l2": top_layer_preview(layers, "importance_l2"),
        "top_importance_semantic": top_layer_preview(layers, "importance_semantic"),
        "top_conflict_raw": top_layer_preview(layers, "conflict_raw"),
    }
    return summary


def append_layer_diagnostic_record(output_dir, record):
    diagnostics_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = os.path.join(diagnostics_dir, "layer_diagnostics.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    summary_path = os.path.join(diagnostics_dir, "layer_diagnostics_summary.json")
    history = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                history = json.load(f).get("history", [])
        except json.JSONDecodeError:
            history = []
    history.append(record["summary"])
    payload = {
        "latest": record["summary"],
        "history": history,
        "jsonl_path": jsonl_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_layer_diagnostic_logging(net_focussr, net_lpips, net_csd, batch, args, global_step, write_output=True):
    stage, _trainable_count = current_lora_train_stage(net_focussr)
    if stage not in {"pixel", "semantic"}:
        raise RuntimeError("no trainable pixel/semantic LoRA params found for layer diagnostics")

    objectives = parse_layer_diagnostic_objectives(args.layer_diagnostic_objectives)
    selected, module_to_param_names = collect_lora_diagnostic_params(net_focussr, stage)
    if not selected:
        raise RuntimeError(f"no trainable {stage} LoRA params selected for layer diagnostics")

    losses, grad_maps = compute_layer_diagnostic_grads(
        net_focussr,
        net_lpips,
        net_csd,
        batch,
        args,
        selected,
        objectives,
    )
    layers, totals = build_layer_diagnostic_rows(selected, module_to_param_names, grad_maps, objectives)
    summary = build_layer_diagnostic_summary(global_step, stage, objectives, losses, layers, totals, selected)
    record = {
        "step": int(global_step),
        "stage": stage,
        "objectives": objectives,
        "losses": losses,
        "summary": summary,
        "layers": layers,
    }
    if write_output:
        append_layer_diagnostic_record(args.output_dir, record)
    return summary


def semantic_delta_consistency_loss(net_focussr, x_src, x_tgt, batch, args):
    if args.consistency_aug != "hflip":
        raise ValueError(f"Unsupported consistency_aug: {args.consistency_aug}")
    if args.consistency_space != "semantic_delta":
        raise ValueError(f"Unsupported consistency_space: {args.consistency_space}")

    x_src_flip = torch.flip(x_src, dims=[-1])

    with torch.no_grad():
        set_active_adapters(net_focussr, PIX_ADAPTERS)
        y_pix, _, _, _ = net_focussr(
            x_src, x_tgt, batch=batch, args=args,
            force_null_text=False, deterministic_vae=True,
        )
        y_pix_flip, _, _, _ = net_focussr(
            x_src_flip, x_tgt, batch=batch, args=args,
            force_null_text=False, deterministic_vae=True,
        )
        y_pix_flip = torch.flip(y_pix_flip, dims=[-1])

    set_active_adapters(net_focussr, semantic_adapter_names_for_training(net_focussr, args))
    y_sem, _, _, _ = net_focussr(
        x_src, x_tgt, batch=batch, args=args,
        force_null_text=False, deterministic_vae=True,
    )
    y_sem_flip, _, _, _ = net_focussr(
        x_src_flip, x_tgt, batch=batch, args=args,
        force_null_text=False, deterministic_vae=True,
    )
    y_sem_flip = torch.flip(y_sem_flip, dims=[-1])

    delta_sem = y_sem - y_pix.detach()
    delta_sem_flip = y_sem_flip - y_pix_flip.detach()
    return F.l1_loss(delta_sem.float(), delta_sem_flip.float(), reduction="mean")


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    load_rank_curriculum_metadata(args)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_focussr = FoCuSSR(args)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_focussr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_focussr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # init CSDLoss model
    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # Set the first trainable stage before accelerator.prepare. DDP records
    # trainable parameters at wrap time, so semantic-only stages must enable
    # their role adapters here rather than only inside the training loop.
    if args.pix_steps <= 0:
        activate_semantic_training(net_focussr, args)
    else:
        net_focussr.unet.set_adapter(PIX_ADAPTERS)
        net_focussr.set_train_pix() # first to remove degradation

    # make the optimizer
    layers_to_opt, optimizer_params, optimizer_metadata = build_lora_optimizer_params(net_focussr, args)
    for key, value in optimizer_metadata.items():
        setattr(args, key, value)
    if accelerator.is_main_process:
        print("====> FoCuS-SR optimizer config")
        for key, value in optimizer_metadata.items():
            print(f"{key}: {value}")
        if getattr(args, "semantic_rank_curriculum_json", None):
            print("====> semantic rank curriculum config")
            print(f"semantic_rank_curriculum_json: {args.semantic_rank_curriculum_json}")
            print(f"semantic_rank_curriculum_num_modules: {getattr(args, 'semantic_rank_curriculum_num_modules', 0)}")
            print(f"semantic_rank_curriculum_budget_relative_error: {getattr(args, 'semantic_rank_curriculum_budget_relative_error', 0.0)}")
        if getattr(args, "pixel_rank_curriculum_json", None):
            print("====> pixel rank curriculum config")
            print(f"pixel_rank_curriculum_json: {args.pixel_rank_curriculum_json}")
            print(f"pixel_rank_curriculum_num_modules: {getattr(args, 'pixel_rank_curriculum_num_modules', 0)}")
            print(f"pixel_rank_curriculum_budget_relative_error: {getattr(args, 'pixel_rank_curriculum_budget_relative_error', 0.0)}")
        if getattr(args, "sequential_semantic_lora_curriculum_json", None):
            curriculum = getattr(net_focussr, "sequential_semantic_lora_curriculum", None) or {}
            print("====> sequential semantic LoRA config")
            print(f"sequential_semantic_lora_curriculum_json: {args.sequential_semantic_lora_curriculum_json}")
            print(f"semantic_train_phase: {args.semantic_train_phase}")
            print(f"stage2a_low_conflict_modules: {len(curriculum.get('stage2a_low_conflict_modules', []) or [])}")
            print(f"stage2b_high_conflict_modules: {len(curriculum.get('stage2b_high_conflict_modules', []) or [])}")
            print(f"detail_modules: {len(curriculum.get('detail_modules', []) or [])}")
            print(f"neutral_modules: {len(curriculum.get('neutral_modules', []) or [])}")
            print(f"safe_modules: {len(curriculum.get('safe_modules', []) or [])}")
        save_hparams(args)

    optimizer = torch.optim.AdamW(optimizer_params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    
    # initialize the dataset
    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    

    # init RAM for text prompt extractor
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    RAM = ram(pretrained='src/ram_pretrain_model/ram_swin_large_14m.pth',
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
    RAM.eval()
    RAM.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    net_focussr, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_focussr, optimizer, dl_train, lr_scheduler
    )
    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = sanitize_tracker_config(args)
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    global_step = 0
    stop_training = False
    lambda_l2 = args.lambda_l2
    lambda_lpips = 0
    lambda_csd = 0
    if args.enable_pixel_stage_perceptual_losses:
        lambda_lpips = args.lambda_lpips
        lambda_csd = args.lambda_csd
    if args.resume_ckpt is not None and args.pix_steps == 10:
        args.pix_steps = 1
    semantic_started = False
    if args.pix_steps <= 0:
        activate_semantic_training(net_focussr, args)
        lambda_l2 = args.lambda_l2
        lambda_lpips = args.lambda_lpips
        lambda_csd = args.lambda_csd
        semantic_started = True
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_focussr):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                # get text prompts from GT
                x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                batch["prompt"] = [f'{each_caption}, {args.pos_prompt_csd}' for each_caption in caption]
                
                if (not semantic_started) and global_step == args.pix_steps:
                    # begin the semantic optimization
                    activate_semantic_training(net_focussr, args)
                    
                    lambda_l2 = args.lambda_l2
                    lambda_lpips = args.lambda_lpips
                    lambda_csd = args.lambda_csd
                    semantic_started = True
                    
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = net_focussr(x_src, x_tgt, batch=batch, args=args)
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * lambda_lpips
                loss = loss_l2 + loss_lpips
                # reg loss
                loss_csd = net_csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args, ) * lambda_csd
                loss = loss + loss_csd
                if args.lambda_consistency > 0 and global_step >= args.pix_steps:
                    loss_consistency = semantic_delta_consistency_loss(net_focussr, x_src, x_tgt, batch, args) * args.lambda_consistency
                    loss = loss + loss_consistency
                else:
                    loss_consistency = loss.new_tensor(0.0)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    diagnostic_summary = None
                    diagnostic_skipped = False
                    if should_log_layer_diagnostics(args, global_step):
                        diagnostic_stage, _ = current_lora_train_stage(net_focussr)
                        diagnostic_skipped = args.resume_ckpt is not None and diagnostic_stage != "semantic"
                        if not diagnostic_skipped:
                            diagnostic_summary = run_layer_diagnostic_logging(
                                net_focussr,
                                net_lpips,
                                net_csd,
                                batch,
                                args,
                                global_step,
                                write_output=accelerator.is_main_process,
                            )
                            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    if accelerator.is_main_process:
                        logs = {}
                        # log all the losses
                        logs["loss_csd"] = loss_csd.detach().item()
                        logs["loss_l2"] = loss_l2.detach().item()
                        logs["loss_lpips"] = loss_lpips.detach().item()
                        logs["loss_consistency"] = loss_consistency.detach().item()
                        if diagnostic_skipped:
                            logs["layer_diagnostic_skipped"] = 1.0
                        if diagnostic_summary is not None:
                            logs["layer_diag_num_layers"] = diagnostic_summary["num_layers"]
                            logs["layer_diag_mean_conflict_relu"] = diagnostic_summary["mean_conflict_relu"] or 0.0
                            logs["layer_diag_mean_conflict_raw"] = diagnostic_summary["mean_conflict_raw"] or 0.0
                            logs["layer_diag_global_conflict_relu"] = diagnostic_summary["global_conflict_relu"] or 0.0
                            logs["layer_diag_global_conflict_raw"] = diagnostic_summary["global_conflict_raw"] or 0.0
                            logs["layer_diag_spearman_semantic_conflict_raw"] = (
                                diagnostic_summary["spearman_importance_semantic_vs_conflict_raw"] or 0.0
                            )
                            logs["layer_diag_spearman_l2_conflict_raw"] = (
                                diagnostic_summary["spearman_importance_l2_vs_conflict_raw"] or 0.0
                            )
                        progress_bar.set_postfix(**logs)

                        # checkpoint the model
                        if global_step % args.checkpointing_steps == 1:
                            outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                            accelerator.unwrap_model(net_focussr).save_model(outf)

                        # test
                        if global_step % args.eval_freq == 1:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                            for step, batch_val in enumerate(dl_val):
                                x_src = batch_val["conditioning_pixel_values"].cuda()
                                x_tgt = batch_val["output_pixel_values"].cuda()
                                x_basename = batch_val["base_name"][0]
                                B, C, H, W = x_src.shape
                                assert B == 1, "Use batch size 1 for eval."
                                with torch.no_grad():
                                    # get text prompts from LR
                                    x_src_ram = ram_transforms(x_src * 0.5 + 0.5)
                                    caption = inference(x_src_ram.to(dtype=torch.float16), RAM)
                                    batch_val["prompt"] = caption
                                    # forward pass
                                    x_tgt_pred, latents_pred, _, _ = accelerator.unwrap_model(net_focussr)(x_src, x_tgt,
                                                                                                          batch=batch_val,
                                                                                                          args=args)
                                    # save the output
                                    output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                    input_image = transforms.ToPILImage()(x_src[0].cpu() * 0.5 + 0.5)
                                    if args.align_method == 'adain':
                                        output_pil = adain_color_fix(target=output_pil, source=input_image)
                                    elif args.align_method == 'wavelet':
                                        output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                                    else:
                                        pass
                                    outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename}")
                                    output_pil.save(outf)
                            gc.collect()
                            torch.cuda.empty_cache()
                            accelerator.log(logs, step=global_step)

                        accelerator.log(logs, step=global_step)

                    if global_step >= args.max_train_steps:
                        if accelerator.is_main_process:
                            final_ckpt = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                            accelerator.unwrap_model(net_focussr).save_model(final_ckpt)
                        stop_training = True

                if stop_training:
                    break
        if stop_training:
            break

if __name__ == "__main__":
    args = parse_args()
    main(args)

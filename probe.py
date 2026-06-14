import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict

import diffusers
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms
from tqdm.auto import tqdm

from model import CSDLoss, FoCuSSR
from src.datasets.dataset import PairedSROnlineTxtDataset
from src.my_utils.training_utils import parse_args


PIX_ADAPTERS = ["default_encoder_pix", "default_decoder_pix", "default_others_pix"]
OBJECTIVE_PAIRS = [
    ("l2", "lpips"),
    ("l2", "csd"),
    ("lpips", "csd"),
]
OBJECTIVES = ["l2", "lpips", "csd"]
OVERLAP_TOP_KS = [10, 20, 50]


def resolve_degradation_file(args):
    if os.path.isabs(args.deg_file_path) or os.path.exists(args.deg_file_path):
        return
    fallback = os.path.join("src", "datasets", args.deg_file_path)
    if os.path.exists(fallback):
        args.deg_file_path = fallback


def setup_ram():
    from ram.models.ram_lora import ram

    ram_transforms = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = ram(
        pretrained="src/ram_pretrain_model/ram_swin_large_14m.pth",
        pretrained_condition=None,
        image_size=384,
        vit="swin_l",
    )
    model.eval()
    model.to("cuda", dtype=torch.float16)
    return model, ram_transforms


@torch.no_grad()
def add_ram_prompts(batch, ram_model, ram_transforms, args):
    from ram import inference_ram as inference

    x_tgt = batch["output_pixel_values"]
    x_tgt_ram = ram_transforms(x_tgt * 0.5 + 0.5)
    captions = inference(x_tgt_ram.to("cuda", dtype=torch.float16), ram_model)
    batch["prompt"] = [f"{caption}, {args.pos_prompt_csd}" for caption in captions]


def move_batch_to_cuda(batch):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.cuda(non_blocking=True)
        else:
            moved[key] = value
    return moved


def should_track_param(name, target_params):
    if "lora" not in name:
        return False
    if target_params == "pix_lora":
        return "pix" in name
    raise ValueError(f"Unsupported LoRA load scope for FoCuS-SR probing: {target_params}")


def zero_selected_grads(selected_params):
    for _, param in selected_params:
        param.grad = None


def global_grad_stats(grads_a, grads_b):
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for name in grads_a:
        ga = grads_a[name]
        gb = grads_b[name]
        if ga is None or gb is None:
            continue
        ga_flat = ga.reshape(-1)
        gb_flat = gb.reshape(-1)
        dot += torch.dot(ga_flat, gb_flat).item()
        norm_a_sq += torch.dot(ga_flat, ga_flat).item()
        norm_b_sq += torch.dot(gb_flat, gb_flat).item()
    norm_a = math.sqrt(max(norm_a_sq, 0.0))
    norm_b = math.sqrt(max(norm_b_sq, 0.0))
    cosine = dot / (norm_a * norm_b + 1e-12)
    return dot, norm_a, norm_b, max(min(cosine, 1.0), -1.0)


def parse_analysis_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run_name", default="unet_weight_probe_b200", type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--probe_space", default="base_unet_weight", choices=["merged_unet_weight", "base_unet_weight"])
    parser.add_argument("--merge_lora_scope", default="pix_lora", choices=["pix_lora"])
    parser.add_argument("--merged_weight_target", default="semantic_lora_targets", choices=["semantic_lora_targets", "all_unet_weights"])
    parser.add_argument("--base_weight_target", default="pixel_lora_targets", choices=["pixel_lora_targets", "all_unet_weights"])
    parser.add_argument("--num_batches", default=200, type=int)
    parser.add_argument("--hist_bins", default=40, type=int)
    parser.add_argument("--deterministic_vae", action="store_true", default=True)
    parser.add_argument("--no_deterministic_vae", dest="deterministic_vae", action="store_false")
    analysis_args, train_argv = parser.parse_known_args()

    args = parse_args(train_argv)
    for key, value in vars(analysis_args).items():
        setattr(args, key, value)
    if args.checkpoint_path is None:
        args.checkpoint_path = args.resume_ckpt
    return args


def sanitize_args(args):
    checkpoint_required = args.probe_space == "merged_unet_weight"
    if checkpoint_required and args.checkpoint_path in (None, "", "None", "none"):
        raise ValueError("checkpoint_path or resume_ckpt is required for merged_unet_weight probing.")
    if checkpoint_required and not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    # FoCuSSR's built-in resume path expects hparams.yml. We load LoRA weights manually
    # so the analysis also works for copied/archived checkpoints without hparams.
    args.resume_ckpt = None
    if not checkpoint_required:
        args.checkpoint_path = None
    args.highquality_dataset_txt_paths = None
    args.prob = 1.0
    args.report_to = "none"
    args.output_dir = args.output_dir or os.path.join(
        "experiments", "layers", args.run_name
    )
    resolve_degradation_file(args)


def load_checkpoint_lora_metadata(args):
    if args.probe_space == "base_unet_weight":
        return None
    sd = torch.load(args.checkpoint_path, map_location="cpu")
    args.lora_rank_unet_pix = int(sd["lora_rank_unet_pix"])
    args.lora_rank_unet_sem = int(sd["lora_rank_unet_sem"])
    args.pixel_rank_curriculum = sd.get("pixel_rank_curriculum")
    args.semantic_rank_curriculum = sd.get("semantic_rank_curriculum")
    return sd


def load_lora_weights_from_checkpoint(net_focussr, checkpoint_sd, load_scope, checkpoint_path):
    state_dict_unet = checkpoint_sd["state_dict_unet"]
    loaded = []
    skipped_scope = []
    missing = []
    shape_mismatch = []

    for name, param in net_focussr.unet.named_parameters():
        if "lora" not in name:
            continue
        if not should_track_param(name, load_scope):
            skipped_scope.append(name)
            continue
        if name not in state_dict_unet:
            missing.append(name)
            continue
        tensor = state_dict_unet[name]
        if tuple(tensor.shape) != tuple(param.shape):
            shape_mismatch.append(
                {
                    "param": name,
                    "checkpoint_shape": list(tensor.shape),
                    "model_shape": list(param.shape),
                }
            )
            continue
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
        loaded.append(name)

    if not loaded:
        raise RuntimeError(f"No LoRA weights were loaded from {checkpoint_path}")

    return {
        "loaded": len(loaded),
        "skipped_scope": len(skipped_scope),
        "missing": missing,
        "shape_mismatch": shape_mismatch,
    }


def semantic_lora_target_modules(net_focussr):
    modules = []
    for attr in (
        "lora_unet_modules_encoder_sem",
        "lora_unet_modules_decoder_sem",
        "lora_unet_others_sem",
    ):
        modules.extend(getattr(net_focussr, attr, []))
    return sorted(set(modules))


def pixel_lora_target_modules(net_focussr):
    modules = []
    for attr in (
        "lora_unet_modules_encoder_pix",
        "lora_unet_modules_decoder_pix",
        "lora_unet_others_pix",
    ):
        modules.extend(getattr(net_focussr, attr, []))
    return sorted(set(modules))


def normalize_unet_weight_module_name(param_name):
    if param_name.endswith(".base_layer.weight"):
        return param_name[: -len(".base_layer.weight")]
    if param_name.endswith(".weight"):
        return param_name[: -len(".weight")]
    return None


def select_unet_weight_params(net_focussr, target_modules=None, all_weights=False):
    target_modules = set(target_modules or [])

    selected = []
    for name, param in net_focussr.unet.named_parameters():
        module_name = normalize_unet_weight_module_name(name)
        select = (
            module_name is not None
            and ".lora_" not in name
            and param.ndim >= 2
            and (all_weights or module_name in target_modules)
        )
        param.requires_grad_(select)
        if select:
            selected.append((name, param))

    return selected


def select_merged_unet_weight_params(net_focussr, target):
    if target == "semantic_lora_targets":
        selected = select_unet_weight_params(net_focussr, target_modules=semantic_lora_target_modules(net_focussr))
    elif target == "all_unet_weights":
        selected = select_unet_weight_params(net_focussr, all_weights=True)
    else:
        raise ValueError(f"Unsupported merged_weight_target: {target}")

    if not selected:
        raise RuntimeError(f"No merged UNet weights selected for merged_weight_target={target}")
    return selected


def prepare_merged_unet_weight_probe(net_focussr, checkpoint_sd, args):
    if args.merge_lora_scope != "pix_lora":
        raise ValueError(f"Unsupported merge_lora_scope for v1: {args.merge_lora_scope}")

    load_report = load_lora_weights_from_checkpoint(
        net_focussr,
        checkpoint_sd,
        "pix_lora",
        args.checkpoint_path,
    )
    target_modules = semantic_lora_target_modules(net_focussr)

    set_weights_and_activate_adapters(net_focussr.unet, PIX_ADAPTERS, [1.0] * len(PIX_ADAPTERS))
    merged_unet = net_focussr.unet.merge_and_unload(
        progressbar=False,
        safe_merge=False,
        adapter_names=PIX_ADAPTERS,
    )
    if merged_unet is not None:
        net_focussr.unet = merged_unet

    selected_params = select_merged_unet_weight_params(net_focussr, args.merged_weight_target)
    lora_params_remaining = sum(1 for name, _ in net_focussr.unet.named_parameters() if "lora" in name)
    load_report.update(
        {
            "probe_space": args.probe_space,
            "merge_lora_scope": args.merge_lora_scope,
            "merged_weight_target": args.merged_weight_target,
            "semantic_lora_target_modules": len(target_modules),
            "selected_merged_weights": len(selected_params),
            "lora_params_remaining_after_merge": lora_params_remaining,
        }
    )
    return selected_params, load_report


def select_base_unet_weight_params(net_focussr, target):
    if target == "pixel_lora_targets":
        selected = select_unet_weight_params(net_focussr, target_modules=pixel_lora_target_modules(net_focussr))
    elif target == "all_unet_weights":
        selected = select_unet_weight_params(net_focussr, all_weights=True)
    else:
        raise ValueError(f"Unsupported base_weight_target: {target}")

    if not selected:
        raise RuntimeError(f"No base UNet weights selected for base_weight_target={target}")
    return selected


def prepare_base_unet_weight_probe(net_focussr, args):
    # Keep the PEFT wrappers only as a convenient way to recover LoRA-eligible
    # module names. The actual probing forward must be the plain base UNet.
    net_focussr.unet.disable_adapters()
    selected_params = select_base_unet_weight_params(net_focussr, args.base_weight_target)
    lora_params_selected = sum(1 for name, param in net_focussr.unet.named_parameters() if "lora" in name and param.requires_grad)
    if lora_params_selected:
        raise RuntimeError(f"Base weight probe selected {lora_params_selected} LoRA params; expected 0.")

    load_report = {
        "probe_space": args.probe_space,
        "base_weight_target": args.base_weight_target,
        "pixel_lora_target_modules": len(pixel_lora_target_modules(net_focussr)),
        "selected_base_weights": len(selected_params),
        "lora_params_selected": lora_params_selected,
        "adapters_disabled": True,
    }
    return selected_params, load_report


def capture_grads(selected_params):
    grads = {}
    for name, param in selected_params:
        if param.grad is None:
            grads[name] = None
        else:
            grads[name] = param.grad.detach().float().cpu().clone()
    return grads


def forward_single_objective(net_focussr, net_lpips, net_csd, batch, args, objective):
    x_src = batch["conditioning_pixel_values"]
    x_tgt = batch["output_pixel_values"]
    y_pred, latents_pred, prompt_embeds, neg_prompt_embeds = net_focussr(
        x_src,
        x_tgt,
        batch=batch,
        args=args,
        force_null_text=False,
        deterministic_vae=args.deterministic_vae,
    )

    if objective == "l2":
        loss = F.mse_loss(y_pred.float(), x_tgt.float(), reduction="mean")
    elif objective == "lpips":
        loss = net_lpips(y_pred.float(), x_tgt.float()).mean()
    elif objective == "csd":
        loss = net_csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args)
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    return loss


def compute_objective_grads(net_focussr, net_lpips, net_csd, batch, args, selected_params):
    losses = {}
    grads = {}
    for objective in ["l2", "lpips", "csd"]:
        zero_selected_grads(selected_params)
        loss = forward_single_objective(net_focussr, net_lpips, net_csd, batch, args, objective)
        loss.backward()
        losses[objective] = loss.detach().item()
        grads[objective] = capture_grads(selected_params)
    zero_selected_grads(selected_params)
    return losses, grads


def pair_key(obj_a, obj_b, field):
    return f"{field}_{obj_a}_{obj_b}"


def compute_pair_stats(grads):
    stats = {}
    for obj_a, obj_b in OBJECTIVE_PAIRS:
        dot, norm_a, norm_b, cosine = global_grad_stats(grads[obj_a], grads[obj_b])
        stats[pair_key(obj_a, obj_b, "dot")] = dot
        stats[pair_key(obj_a, obj_b, "grad_norm_a")] = norm_a
        stats[pair_key(obj_a, obj_b, "grad_norm_b")] = norm_b
        stats[pair_key(obj_a, obj_b, "cos")] = cosine
    return stats


def update_layer_stats(layer_stats, grads):
    for obj_a, obj_b in OBJECTIVE_PAIRS:
        for name in grads[obj_a]:
            ga = grads[obj_a][name]
            gb = grads[obj_b][name]
            if ga is None or gb is None:
                continue
            ga_flat = ga.reshape(-1)
            gb_flat = gb.reshape(-1)
            key = f"{obj_a}_{obj_b}"
            layer_stats[name][key]["dot"] += torch.dot(ga_flat, gb_flat).item()
            layer_stats[name][key]["norm_a_sq"] += torch.dot(ga_flat, ga_flat).item()
            layer_stats[name][key]["norm_b_sq"] += torch.dot(gb_flat, gb_flat).item()
            layer_stats[name][key]["num_batches"] += 1


def update_importance_stats(importance_stats, grads):
    for objective in OBJECTIVES:
        for name, grad in grads[objective].items():
            if grad is None:
                continue
            grad_flat = grad.reshape(-1)
            importance_stats[name][objective]["norm_sq"] += torch.dot(grad_flat, grad_flat).item()
            importance_stats[name][objective]["num_batches"] += 1


def summarize_pair(records, obj_a, obj_b):
    key = pair_key(obj_a, obj_b, "cos")
    cosines = np.array([record[key] for record in records], dtype=np.float64)
    return {
        f"mean_{key}": float(np.mean(cosines)),
        f"median_{key}": float(np.median(cosines)),
        f"negative_ratio_{obj_a}_{obj_b}": float(np.mean(cosines < 0.0)),
        f"strong_conflict_ratio_{obj_a}_{obj_b}": float(np.mean(cosines < -0.1)),
        f"weak_alignment_ratio_{obj_a}_{obj_b}": float(np.mean((cosines >= -0.05) & (cosines <= 0.05))),
    }


def summarize_records(records, args, load_report):
    summary = {
        "run_name": args.run_name,
        "num_batches": len(records),
        "probe_space": args.probe_space,
        "merge_lora_scope": args.merge_lora_scope,
        "merged_weight_target": args.merged_weight_target,
        "base_weight_target": args.base_weight_target,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_load_report": load_report,
        "dataset_txt": args.dataset_txt_paths,
        "pretrained_model_path": args.pretrained_model_path,
        "resume_ckpt_used_by_focussr": None,
        "objectives": OBJECTIVES,
        "deterministic_vae": args.deterministic_vae,
        "train_batch_size": args.train_batch_size,
        "resolution_ori": args.resolution_ori,
        "resolution_tgt": args.resolution_tgt,
        "deg_file_path": args.deg_file_path,
    }
    for obj_a, obj_b in OBJECTIVE_PAIRS:
        summary.update(summarize_pair(records, obj_a, obj_b))
    for objective in OBJECTIVES:
        values = np.array([record[f"loss_{objective}"] for record in records], dtype=np.float64)
        summary[f"mean_loss_{objective}"] = float(np.mean(values))
    return summary


def build_layer_summary_rows(layer_stats):
    rows = []
    for name, pair_map in layer_stats.items():
        row = {"param": name}
        for pair_name, stats in pair_map.items():
            norm_a = math.sqrt(max(stats["norm_a_sq"], 0.0))
            norm_b = math.sqrt(max(stats["norm_b_sq"], 0.0))
            cosine = stats["dot"] / (norm_a * norm_b + 1e-12)
            row[f"cos_{pair_name}"] = max(min(cosine, 1.0), -1.0)
            row[f"dot_{pair_name}"] = stats["dot"]
            row[f"num_batches_{pair_name}"] = stats["num_batches"]
        rows.append(row)
    rows.sort(key=lambda item: item.get("cos_l2_csd", 0.0))
    return rows


def write_layer_summary(path, layer_stats):
    rows = build_layer_summary_rows(layer_stats)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def rank_rows(rows, key, reverse=True, rank_key=None):
    sorted_rows = sorted(rows, key=lambda item: item.get(key, 0.0), reverse=reverse)
    rank_key = rank_key or f"rank_{key}"
    for rank, row in enumerate(sorted_rows, start=1):
        row[rank_key] = rank


def build_layer_importance_rows(layer_stats, importance_stats):
    rows_by_name = {row["param"]: row for row in build_layer_summary_rows(layer_stats)}
    totals = {}
    for objective in OBJECTIVES:
        total = 0.0
        for objective_map in importance_stats.values():
            total += math.sqrt(max(objective_map[objective]["norm_sq"], 0.0))
        totals[objective] = total

    for name, objective_map in importance_stats.items():
        row = rows_by_name.setdefault(name, {"param": name})
        for objective in OBJECTIVES:
            grad_norm = math.sqrt(max(objective_map[objective]["norm_sq"], 0.0))
            row[f"grad_norm_{objective}"] = grad_norm
            row[f"importance_{objective}"] = grad_norm / (totals[objective] + 1e-12)
            row[f"num_batches_importance_{objective}"] = objective_map[objective]["num_batches"]
        row["conflict_score_l2_csd"] = -float(row.get("cos_l2_csd", 0.0))

    rows = list(rows_by_name.values())
    rank_rows(rows, "importance_l2", reverse=True, rank_key="rank_importance_l2")
    rank_rows(rows, "importance_lpips", reverse=True, rank_key="rank_importance_lpips")
    rank_rows(rows, "importance_csd", reverse=True, rank_key="rank_importance_csd")
    rank_rows(rows, "conflict_score_l2_csd", reverse=True, rank_key="rank_conflict_l2_csd")
    rows.sort(key=lambda item: item.get("rank_conflict_l2_csd", len(rows) + 1))
    return rows, totals


def spearman_correlation(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2 or ys.size < 2:
        return None
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return None
    x_order = np.argsort(np.argsort(xs))
    y_order = np.argsort(np.argsort(ys))
    return float(np.corrcoef(x_order, y_order)[0, 1])


def build_overlap_summary(layer_importance_rows, importance_totals, top_ks=None):
    top_ks = top_ks or OVERLAP_TOP_KS
    params_by_conflict = [
        row["param"]
        for row in sorted(layer_importance_rows, key=lambda item: item.get("conflict_score_l2_csd", 0.0), reverse=True)
    ]

    overlap = {}
    for objective in OBJECTIVES:
        params_by_importance = [
            row["param"]
            for row in sorted(layer_importance_rows, key=lambda item: item.get(f"importance_{objective}", 0.0), reverse=True)
        ]
        objective_overlap = {}
        for k in top_ks:
            kk = min(k, len(layer_importance_rows))
            conflict_set = set(params_by_conflict[:kk])
            importance_set = set(params_by_importance[:kk])
            shared = sorted(conflict_set & importance_set)
            objective_overlap[f"top_{k}"] = {
                "k_effective": kk,
                "overlap_count": len(shared),
                "overlap_ratio": float(len(shared) / kk) if kk else 0.0,
                "shared_params": shared,
            }
        overlap[objective] = objective_overlap

    conflict_scores = [row.get("conflict_score_l2_csd", 0.0) for row in layer_importance_rows]
    correlations = {}
    for objective in OBJECTIVES:
        importances = [row.get(f"importance_{objective}", 0.0) for row in layer_importance_rows]
        correlations[f"spearman_conflict_score_l2_csd_vs_importance_{objective}"] = spearman_correlation(
            conflict_scores, importances
        )

    return {
        "num_layers": len(layer_importance_rows),
        "top_ks": top_ks,
        "importance_totals": importance_totals,
        "overlap": overlap,
        "correlations": correlations,
    }


def write_layer_importance_outputs(output_dir, layer_stats, importance_stats):
    rows, totals = build_layer_importance_rows(layer_stats, importance_stats)
    layer_importance_path = os.path.join(output_dir, "layer_importance.json")
    with open(layer_importance_path, "w") as f:
        json.dump(rows, f, indent=2)

    overlap_summary = build_overlap_summary(rows, totals)
    overlap_path = os.path.join(output_dir, "overlap_with_conflict.json")
    with open(overlap_path, "w") as f:
        json.dump(overlap_summary, f, indent=2)
    return rows, overlap_summary


def write_histograms(output_dir, records, hist_bins):
    for obj_a, obj_b in OBJECTIVE_PAIRS:
        key = pair_key(obj_a, obj_b, "cos")
        cosines = np.array([record[key] for record in records], dtype=np.float64)
        counts, edges = np.histogram(cosines, bins=hist_bins, range=(-1.0, 1.0))
        path = os.path.join(output_dir, f"cosine_hist_{obj_a}_{obj_b}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["bin_left", "bin_right", "count"])
            writer.writeheader()
            for idx, count in enumerate(counts):
                writer.writerow(
                    {
                        "bin_left": float(edges[idx]),
                        "bin_right": float(edges[idx + 1]),
                        "count": int(count),
                    }
                )


def nested_pair_stats_factory():
    return defaultdict(lambda: {"dot": 0.0, "norm_a_sq": 0.0, "norm_b_sq": 0.0, "num_batches": 0})


def main():
    args = parse_analysis_args()
    sanitize_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint LoRA gradient conflict analysis requires CUDA.")

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    checkpoint_sd = load_checkpoint_lora_metadata(args)
    os.makedirs(args.output_dir, exist_ok=True)

    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    net_focussr = FoCuSSR(args)
    net_focussr.unet.train()
    net_focussr.text_encoder.requires_grad_(False)
    net_focussr.vae_fix.requires_grad_(False)
    if args.probe_space == "merged_unet_weight":
        selected_params, load_report = prepare_merged_unet_weight_probe(net_focussr, checkpoint_sd, args)
    elif args.probe_space == "base_unet_weight":
        selected_params, load_report = prepare_base_unet_weight_probe(net_focussr, args)
    else:
        raise ValueError(f"Unsupported probe_space: {args.probe_space}")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_focussr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)
    net_lpips.eval()

    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)
    net_csd.eval()

    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    ram_model, ram_transforms = setup_ram()

    config = dict(vars(args))
    config["checkpoint_load_report"] = load_report
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    records = []
    layer_stats = defaultdict(nested_pair_stats_factory)
    importance_stats = defaultdict(
        lambda: defaultdict(lambda: {"norm_sq": 0.0, "num_batches": 0})
    )
    records_path = os.path.join(args.output_dir, "records.jsonl")

    with open(records_path, "w") as record_file:
        progress = tqdm(total=args.num_batches, desc="Checkpoint gradient conflict batches")
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break
            batch = move_batch_to_cuda(batch)
            add_ram_prompts(batch, ram_model, ram_transforms, args)

            losses, grads = compute_objective_grads(net_focussr, net_lpips, net_csd, batch, args, selected_params)
            pair_stats = compute_pair_stats(grads)
            update_layer_stats(layer_stats, grads)
            update_importance_stats(importance_stats, grads)

            record = {
                "batch_idx": batch_idx,
                "loss_l2": losses["l2"],
                "loss_lpips": losses["lpips"],
                "loss_csd": losses["csd"],
                **pair_stats,
            }
            records.append(record)
            record_file.write(json.dumps(record) + "\n")
            record_file.flush()

            progress.set_postfix(
                l2_csd=f"{record['cos_l2_csd']:.4f}",
                neg=f"{np.mean([r['cos_l2_csd'] < 0 for r in records]):.2f}",
            )
            progress.update(1)
        progress.close()

    if not records:
        raise RuntimeError("No records were produced. Check dataset path and num_batches.")

    summary = summarize_records(records, args, load_report)
    write_layer_summary(os.path.join(args.output_dir, "layer_summary.json"), layer_stats)
    _, overlap_summary = write_layer_importance_outputs(args.output_dir, layer_stats, importance_stats)
    summary["layer_importance"] = {
        "num_layers": overlap_summary["num_layers"],
        "importance_totals": overlap_summary["importance_totals"],
        "correlations": overlap_summary["correlations"],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_histograms(args.output_dir, records, args.hist_bins)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

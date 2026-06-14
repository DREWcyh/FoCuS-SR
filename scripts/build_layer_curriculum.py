#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def parse_csv_numbers(value, cast):
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not parts:
        raise ValueError(f"empty comma-separated value: {value}")
    return [cast(item) for item in parts]


def module_from_param(param_name):
    if ".lora_" in param_name:
        return param_name.split(".lora_", 1)[0]
    if param_name.endswith(".base_layer.weight"):
        return param_name[: -len(".base_layer.weight")]
    if param_name.endswith(".weight"):
        return param_name[: -len(".weight")]
    return None


def normalize_scores(raw_scores):
    if not raw_scores:
        return {}
    values = list(raw_scores.values())
    min_value = min(values)
    max_value = max(values)
    denom = max(max_value - min_value, 1e-12)
    return {key: (value - min_value) / denom for key, value in raw_scores.items()}


def rank_groups_by_percentile(scores, high_ratio=0.30, low_ratio=0.30):
    ordered = sorted(scores.items(), key=lambda item: item[1])
    n_items = len(ordered)
    low_count = int(math.floor(n_items * low_ratio))
    high_count = int(math.ceil(n_items * high_ratio))
    groups = {}
    for idx, (name, _score) in enumerate(ordered):
        if idx < low_count:
            groups[name] = "low"
        elif idx >= n_items - high_count:
            groups[name] = "high"
        else:
            groups[name] = "mid"
    return groups


def aggregate_scores(layer_records):
    by_module = defaultdict(list)
    for record in layer_records:
        module = module_from_param(record.get("param", ""))
        if module is not None:
            by_module[module].append(record)

    importance_raw = {}
    conflict_raw = {}
    details = {}
    for module, records in by_module.items():
        importance_lpips = sum(float(record.get("importance_lpips", 0.0)) for record in records)
        importance_csd = sum(float(record.get("importance_csd", 0.0)) for record in records)
        conflict = max(max(0.0, -float(record.get("cos_l2_csd", 0.0))) for record in records)
        importance_raw[module] = importance_lpips + importance_csd
        conflict_raw[module] = conflict
        details[module] = {
            "importance_lpips": importance_lpips,
            "importance_csd": importance_csd,
            "importance_semantic_raw": importance_raw[module],
            "conflict_l2_csd_raw": conflict,
            "num_weight_tensors": len(records),
        }
    return importance_raw, conflict_raw, details


def module_costs_from_checkpoint(checkpoint_path, adapter_pattern="sem", baseline_rank=4):
    sd = torch.load(checkpoint_path, map_location="cpu")
    state_dict = sd.get("state_dict_unet", sd)
    modules = defaultdict(dict)

    for name, tensor in state_dict.items():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if adapter_pattern and adapter_pattern not in name:
            continue
        module = module_from_param(name)
        if module is None:
            continue
        if ".lora_A." in name:
            modules[module]["A"] = tensor
        elif ".lora_B." in name:
            modules[module]["B"] = tensor

    costs = {}
    for module, tensors in modules.items():
        if "A" not in tensors or "B" not in tensors:
            continue
        count_at_baseline = int(tensors["A"].numel() + tensors["B"].numel())
        costs[module] = count_at_baseline / float(baseline_rank)
    return costs


def budget(rank_pattern, costs):
    return sum(float(rank_pattern[module]) * costs[module] for module in rank_pattern)


def correct_budget(rank_pattern, costs, scores, baseline_budget, tolerance):
    rank_pattern = dict(rank_pattern)
    modules_by_importance_asc = sorted(rank_pattern, key=lambda name: scores[name])
    modules_by_importance_desc = list(reversed(modules_by_importance_asc))

    def rel_error():
        return abs(budget(rank_pattern, costs) - baseline_budget) / max(baseline_budget, 1e-12)

    for from_rank, to_rank in ((8, 6), (6, 4), (4, 3), (3, 2)):
        while budget(rank_pattern, costs) > baseline_budget * (1.0 + tolerance):
            candidates = [name for name in modules_by_importance_asc if rank_pattern[name] == from_rank]
            if not candidates:
                break
            rank_pattern[candidates[0]] = to_rank
        if rel_error() <= tolerance:
            return rank_pattern

    for from_rank, to_rank in ((4, 6), (2, 3), (3, 4), (6, 8)):
        while budget(rank_pattern, costs) < baseline_budget * (1.0 - tolerance):
            candidates = [name for name in modules_by_importance_desc if rank_pattern[name] == from_rank]
            if not candidates:
                break
            rank_pattern[candidates[0]] = to_rank
        if rel_error() <= tolerance:
            return rank_pattern

    allowed_ranks = [2, 3, 4, 5, 6, 8]
    for _ in range(10000):
        current_error = rel_error()
        if current_error <= tolerance:
            break
        current_budget = budget(rank_pattern, costs)
        best_move = None
        best_error = current_error
        for module in rank_pattern:
            current_rank = rank_pattern[module]
            possible = (
                [rank for rank in allowed_ranks if rank < current_rank]
                if current_budget > baseline_budget
                else [rank for rank in allowed_ranks if rank > current_rank]
            )
            for new_rank in possible:
                old_rank = rank_pattern[module]
                rank_pattern[module] = new_rank
                candidate_error = rel_error()
                rank_pattern[module] = old_rank
                if candidate_error < best_error:
                    best_error = candidate_error
                    best_move = (module, new_rank)
        if best_move is None:
            break
        rank_pattern[best_move[0]] = best_move[1]
    return rank_pattern


def top_conflict_split(conflict_scores, top_ratio):
    ordered = sorted(conflict_scores.items(), key=lambda item: item[1])
    n_items = len(ordered)
    high_count = int(math.ceil(n_items * top_ratio))
    high = {name for name, _score in ordered[n_items - high_count:]} if high_count > 0 else set()
    low = set(conflict_scores) - high
    return sorted(low), sorted(high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance_json", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline_rank", type=int, default=4)
    parser.add_argument("--rank_values", type=str, default="8,4,2")
    parser.add_argument("--conflict_top_ratio", type=float, default=0.30)
    parser.add_argument("--rank_high_ratio", type=float, default=0.30)
    parser.add_argument("--rank_low_ratio", type=float, default=0.30)
    parser.add_argument("--budget_tolerance", type=float, default=0.005)
    parser.add_argument("--adapter_pattern", type=str, default="sem")
    parser.add_argument("--alpha_scale", type=float, default=2.0)
    args = parser.parse_args()

    rank_high, rank_mid, rank_low = parse_csv_numbers(args.rank_values, int)
    with open(args.importance_json, "r", encoding="utf-8") as f:
        layer_records = json.load(f)

    importance_raw, conflict_raw, module_details = aggregate_scores(layer_records)
    importance_scores = normalize_scores(importance_raw)
    conflict_scores = normalize_scores(conflict_raw)
    costs = module_costs_from_checkpoint(args.checkpoint_path, args.adapter_pattern, args.baseline_rank)

    common_modules = sorted(set(importance_scores) & set(conflict_scores) & set(costs))
    if not common_modules:
        raise RuntimeError("no overlapping semantic LoRA modules between importance JSON and checkpoint")

    importance_scores = {name: importance_scores[name] for name in common_modules}
    conflict_scores = {name: conflict_scores[name] for name in common_modules}
    costs = {name: costs[name] for name in common_modules}
    module_details = {name: module_details[name] for name in common_modules}

    stage2a_modules, stage2b_modules = top_conflict_split(conflict_scores, args.conflict_top_ratio)

    rank_group = rank_groups_by_percentile(
        importance_scores,
        high_ratio=args.rank_high_ratio,
        low_ratio=args.rank_low_ratio,
    )
    group_to_rank = {"high": rank_high, "mid": rank_mid, "low": rank_low}
    rank_pattern = {name: group_to_rank[rank_group[name]] for name in common_modules}

    baseline_budget = sum(args.baseline_rank * costs[name] for name in common_modules)
    rank_pattern = correct_budget(
        rank_pattern,
        costs,
        importance_scores,
        baseline_budget,
        args.budget_tolerance,
    )
    new_budget = budget(rank_pattern, costs)
    budget_relative_error = abs(new_budget - baseline_budget) / max(baseline_budget, 1e-12)
    if budget_relative_error > args.budget_tolerance:
        raise RuntimeError(
            f"budget correction failed: relative error {budget_relative_error:.6f} > {args.budget_tolerance:.6f}"
        )

    alpha_pattern = {name: int(round(rank * args.alpha_scale)) for name, rank in rank_pattern.items()}
    output = {
        "format": "safe_to_risk_rank_curriculum_v1",
        "importance_json": args.importance_json,
        "checkpoint_path": args.checkpoint_path,
        "baseline_rank": args.baseline_rank,
        "rank_values": {"high": rank_high, "mid": rank_mid, "low": rank_low},
        "rank_high_ratio": args.rank_high_ratio,
        "rank_low_ratio": args.rank_low_ratio,
        "conflict_top_ratio": args.conflict_top_ratio,
        "score_definition": {
            "semantic_importance_score": "normalize(sum_A_B(importance_lpips + importance_csd))",
            "semantic_conflict_score": "normalize(max_A_B(max(0, -cos_l2_csd)))",
            "stage_split": "stage2b_high_conflict=top conflict_top_ratio by conflict; stage2a_low_conflict=remaining",
            "rank_allocation": "rank is assigned by semantic importance only",
            "pruning": "disabled; all semantic-LoRA-eligible modules are kept",
        },
        "stage2a_low_conflict_modules": stage2a_modules,
        "stage2b_high_conflict_modules": stage2b_modules,
        "detail_modules": stage2a_modules,
        "neutral_modules": [],
        "safe_modules": stage2b_modules,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "module_importance_score": importance_scores,
        "module_conflict_score": conflict_scores,
        "rank_group": rank_group,
        "module_cost_per_rank": costs,
        "module_details": module_details,
        "baseline_semantic_params": baseline_budget,
        "new_semantic_params": new_budget,
        "budget_relative_error": budget_relative_error,
        "num_modules": len(common_modules),
        "stage2a_low_conflict_count": len(stage2a_modules),
        "stage2b_high_conflict_count": len(stage2b_modules),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "num_modules": len(common_modules),
        "stage2a_low_conflict_count": len(stage2a_modules),
        "stage2b_high_conflict_count": len(stage2b_modules),
        "baseline_semantic_params": baseline_budget,
        "new_semantic_params": new_budget,
        "budget_relative_error": budget_relative_error,
    }, indent=2))


if __name__ == "__main__":
    main()

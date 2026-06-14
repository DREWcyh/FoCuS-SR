#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def read_paths(path, max_images=None):
    paths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                paths.append(item)
            if max_images is not None and len(paths) >= max_images:
                break
    return paths


def image_tensor(path, device):
    image = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor


def compute_musiq(paths, device):
    import pyiqa

    metric = pyiqa.create_metric("musiq", device=device).to(device)
    rows = []
    for path in tqdm(paths, desc="MUSIQ"):
        try:
            with torch.no_grad():
                score = float(metric(image_tensor(path, device)).detach().cpu().item())
            rows.append({"path": path, "musiq": score, "error": None})
        except Exception as exc:
            rows.append({"path": path, "musiq": None, "error": str(exc)})
    return rows


def compute_clip_embeddings(paths, device, model_name, pretrained, batch_size):
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("open_clip is required for CLIP diversity selection") from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()

    embeddings = []
    valid_paths = []
    batch = []
    batch_paths = []
    for path in tqdm(paths, desc="CLIP"):
        try:
            image = Image.open(path).convert("RGB")
            batch.append(preprocess(image))
            batch_paths.append(path)
        except Exception:
            continue
        if len(batch) >= batch_size:
            with torch.no_grad():
                tensor = torch.stack(batch).to(device)
                feat = model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            embeddings.append(feat.detach().cpu().float().numpy())
            valid_paths.extend(batch_paths)
            batch, batch_paths = [], []

    if batch:
        with torch.no_grad():
            tensor = torch.stack(batch).to(device)
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        embeddings.append(feat.detach().cpu().float().numpy())
        valid_paths.extend(batch_paths)

    if not embeddings:
        raise RuntimeError("no CLIP embeddings were computed")
    return valid_paths, np.concatenate(embeddings, axis=0)


def cluster_embeddings(embeddings, cluster_count, seed):
    if len(embeddings) == 1:
        return np.zeros(1, dtype=np.int64)
    k = min(cluster_count, len(embeddings))
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        return numpy_kmeans(embeddings, k, seed)
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=1024)
    return model.fit_predict(embeddings)


def numpy_kmeans(embeddings, cluster_count, seed, max_iter=50):
    rng = np.random.default_rng(seed)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    indices = rng.choice(len(embeddings), size=cluster_count, replace=False)
    centers = embeddings[indices].copy()
    labels = np.zeros(len(embeddings), dtype=np.int64)

    for _ in range(max_iter):
        distances = ((embeddings[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1).astype(np.int64)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(cluster_count):
            members = embeddings[labels == cluster_id]
            if len(members) == 0:
                centers[cluster_id] = embeddings[rng.integers(0, len(embeddings))]
            else:
                centers[cluster_id] = members.mean(axis=0)
    return labels


def balanced_select(cluster_rows, target_count):
    clusters = defaultdict(list)
    for row in cluster_rows:
        clusters[int(row["cluster"])].append(row)
    for rows in clusters.values():
        rows.sort(key=lambda item: item["musiq"], reverse=True)

    selected = []
    cluster_ids = sorted(clusters)
    cursor = {cluster_id: 0 for cluster_id in cluster_ids}
    while True:
        progressed = False
        for cluster_id in cluster_ids:
            idx = cursor[cluster_id]
            if idx >= len(clusters[cluster_id]):
                continue
            selected.append(clusters[cluster_id][idx])
            cursor[cluster_id] += 1
            progressed = True
            if target_count is not None and len(selected) >= target_count:
                return selected
        if not progressed:
            return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_txt", default="preset/gt_all_path.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_txt", default=None)
    parser.add_argument("--musiq_threshold", type=float, default=78.0)
    parser.add_argument("--cluster_count", type=int, default=50)
    parser.add_argument("--target_count", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--clip_model", default="ViT-B-32")
    parser.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--clip_batch_size", type=int, default=32)
    parser.add_argument("--skip_clip", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_txt = Path(args.output_txt) if args.output_txt else output_dir / "gt_hq_clip_musiq_path.txt"

    paths = read_paths(args.input_txt, args.max_images)
    if not paths:
        raise RuntimeError(f"no image paths found in {args.input_txt}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    musiq_rows = compute_musiq(paths, device)
    scores_path = output_dir / "scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as f:
        for row in musiq_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    candidates = [row for row in musiq_rows if row["musiq"] is not None and row["musiq"] > args.musiq_threshold]
    if not candidates:
        raise RuntimeError(
            f"no images passed MUSIQ threshold {args.musiq_threshold}; "
            f"inspect {scores_path} or lower --musiq_threshold"
        )

    if args.skip_clip:
        candidates = sorted(candidates, key=lambda item: item["musiq"], reverse=True)
        valid_paths = [row["path"] for row in candidates]
        cluster_rows = [
            {
                "path": row["path"],
                "musiq": float(row["musiq"]),
                "cluster": 0,
            }
            for row in candidates
        ]
        cluster_count_used = 1
    else:
        candidate_paths = [row["path"] for row in candidates]
        path_to_musiq = {row["path"]: row["musiq"] for row in candidates}
        valid_paths, embeddings = compute_clip_embeddings(
            candidate_paths,
            device,
            args.clip_model,
            args.clip_pretrained,
            args.clip_batch_size,
        )
        labels = cluster_embeddings(embeddings, args.cluster_count, args.seed)
        cluster_rows = [
            {
                "path": path,
                "musiq": float(path_to_musiq[path]),
                "cluster": int(label),
            }
            for path, label in zip(valid_paths, labels)
        ]
        cluster_count_used = int(len(set(labels.tolist())))
    selected = balanced_select(cluster_rows, args.target_count)
    if not selected:
        raise RuntimeError("balanced selection produced no images")

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(row["path"] for row in selected) + "\n", encoding="utf-8")

    cluster_payload = {
        "input_txt": args.input_txt,
        "musiq_threshold": args.musiq_threshold,
        "cluster_count_requested": args.cluster_count,
        "cluster_count_used": cluster_count_used,
        "target_count": args.target_count,
        "selected_count": len(selected),
        "clip_skipped": bool(args.skip_clip),
        "clusters": cluster_rows,
    }
    (output_dir / "clusters.json").write_text(
        json.dumps(cluster_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "input_txt": args.input_txt,
        "output_txt": str(output_txt),
        "num_input_images": len(paths),
        "num_musiq_valid": sum(row["musiq"] is not None for row in musiq_rows),
        "num_hq_candidates": len(candidates),
        "num_clip_valid": len(valid_paths),
        "num_selected": len(selected),
        "musiq_threshold": args.musiq_threshold,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "clip_skipped": bool(args.skip_clip),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

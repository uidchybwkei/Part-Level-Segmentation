import argparse
import base64
import io
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from torchvision import transforms

from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator
from tasks.interactive_idino_m2m_auto import show_anns


def parse_option():
    parser = argparse.ArgumentParser("SemanticSAM FastAPI WebUI", add_help=False)
    parser.add_argument("--ckpt", default="swinl_only_sam_many2many.pth")
    parser.add_argument("--model_type", default="L", choices=["L", "T"])
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


args = parse_option()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"[FastAPI WebUI] Loading model type={args.model_type}, ckpt={args.ckpt}", flush=True)
model = build_semantic_sam(model_type=args.model_type, ckpt=args.ckpt)
print("[FastAPI WebUI] Model loaded", flush=True)

app = FastAPI()
infer_lock = threading.Lock()


def image_to_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def read_upload_to_pil(upload_bytes):
    return Image.open(io.BytesIO(upload_bytes)).convert("RGB")


def prepare_image_from_pil(pil_image, short_edge=640):
    t = transforms.Compose([transforms.Resize(short_edge, interpolation=Image.BICUBIC)])
    image_ori = np.asarray(t(pil_image.convert("RGB")))
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
    return image_ori, images


def render_auto_masks(image_ori, anns):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image_ori)
    show_anns(anns)
    plt.axis("off")
    fig.canvas.draw()
    image = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return image


def small_first_cut_masks(anns, min_mask_area=100, min_visible_ratio=0.15):
    sorted_anns = sorted(anns, key=lambda ann: ann["area"])
    occupied = None
    cut_anns = []
    for ann in sorted_anns:
        mask = ann["segmentation"].astype(bool)
        if occupied is None:
            occupied = np.zeros_like(mask, dtype=bool)
        visible = np.logical_and(mask, np.logical_not(occupied))
        visible_area = int(visible.sum())
        original_area = int(mask.sum())
        if original_area == 0:
            continue
        if visible_area < int(min_mask_area):
            continue
        if visible_area / original_area < float(min_visible_ratio):
            continue
        new_ann = dict(ann)
        new_ann["segmentation"] = visible
        new_ann["area"] = visible_area
        cut_anns.append(new_ann)
        occupied = np.logical_or(occupied, visible)
    return cut_anns


def hierarchy_cut_masks(anns, min_mask_area=100, min_visible_ratio=0.15, contain_thresh=0.85):
    filtered = []
    for idx, ann in enumerate(anns):
        mask = ann["segmentation"].astype(bool)
        area = int(mask.sum())
        if area < int(min_mask_area):
            continue
        new_ann = dict(ann)
        new_ann["segmentation"] = mask
        new_ann["area"] = area
        new_ann["_source_index"] = idx
        filtered.append(new_ann)

    if not filtered:
        return []

    areas = np.array([ann["area"] for ann in filtered], dtype=np.float64)
    masks = [ann["segmentation"] for ann in filtered]
    parents = [-1 for _ in filtered]
    children = [[] for _ in filtered]

    for child_idx, child_mask in enumerate(masks):
        child_area = areas[child_idx]
        best_parent = -1
        best_parent_area = float("inf")
        for parent_idx, parent_mask in enumerate(masks):
            if parent_idx == child_idx or areas[parent_idx] <= child_area:
                continue
            intersection = np.logical_and(child_mask, parent_mask).sum()
            containment = float(intersection) / float(child_area)
            if containment >= float(contain_thresh) and areas[parent_idx] < best_parent_area:
                best_parent = parent_idx
                best_parent_area = areas[parent_idx]
        parents[child_idx] = best_parent
        if best_parent >= 0:
            children[best_parent].append(child_idx)

    cut_anns = []
    for idx, ann in enumerate(filtered):
        visible = ann["segmentation"].copy()
        if children[idx]:
            child_union = np.zeros_like(visible, dtype=bool)
            for child_idx in children[idx]:
                child_union = np.logical_or(child_union, filtered[child_idx]["segmentation"])
            visible = np.logical_and(visible, np.logical_not(child_union))

        visible_area = int(visible.sum())
        if visible_area < int(min_mask_area):
            continue
        if visible_area / float(ann["area"]) < float(min_visible_ratio):
            continue

        new_ann = dict(ann)
        new_ann["segmentation"] = visible
        new_ann["area"] = visible_area
        new_ann["parent_index"] = parents[idx]
        new_ann["child_count"] = len(children[idx])
        cut_anns.append(new_ann)

    return cut_anns


def mask_to_svg_paths(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        points = contour.reshape(-1, 2)
        d = "M " + " L ".join(f"{int(x)} {int(y)}" for x, y in points) + " Z"
        paths.append(d)
    return paths


def mask_to_data_url(mask):
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return image_to_data_url(mask_img.convert("RGB"))


def render_hover_html(image_ori, anns):
    height, width = image_ori.shape[:2]
    base_image = image_to_data_url(Image.fromarray(image_ori))
    path_items = []
    palette = [
        "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6", "#06b6d4",
        "#3b82f6", "#6366f1", "#8b5cf6", "#d946ef", "#ec4899", "#f43f5e",
    ]
    for idx, ann in enumerate(sorted(anns, key=lambda item: item["area"], reverse=True)):
        color = palette[idx % len(palette)]
        score = float(ann.get("predicted_iou", 0.0))
        stability = float(ann.get("stability_score", 0.0))
        child_count = int(ann.get("child_count", 0))
        title = f"mask {idx} | area={int(ann['area'])} | iou={score:.3f} | stability={stability:.3f} | children={child_count}"
        mask_preview = mask_to_data_url(ann["segmentation"])
        for path in mask_to_svg_paths(ann["segmentation"]):
            path_items.append(
                f'<path class="mask-path" d="{path}" fill="{color}" '
                f'data-title="{title}" data-mask="{mask_preview}"><title>{title}</title></path>'
            )
    return f"""
<div class="mask-viewer" style="width:{width}px;max-width:100%;">
  <img src="{base_image}" />
  <svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">
    {''.join(path_items)}
  </svg>
</div>
<div class="mask-preview-panel">
  <div class="mask-preview-title">Hover a mask to preview its binary mask</div>
  <img class="mask-preview-img" alt="Hovered mask preview" />
</div>
"""


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Semantic-SAM Web UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f6f7fb; color: #111827; }
    .card { background: white; border-radius: 14px; padding: 20px; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(0,0,0,.08); }
    .row { display: grid; grid-template-columns: 360px 1fr; gap: 20px; align-items: start; }
    label { display: block; margin-top: 12px; font-weight: 600; }
    input, select, button { margin-top: 6px; padding: 8px; width: 100%; box-sizing: border-box; }
    button { cursor: pointer; background: #2563eb; color: white; border: 0; border-radius: 8px; font-weight: 700; margin-top: 16px; }
    img { max-width: 100%; border-radius: 10px; border: 1px solid #e5e7eb; background: #fff; }
    .mask-viewer { position: relative; border-radius: 10px; overflow: hidden; border: 1px solid #e5e7eb; background: #fff; }
    .mask-viewer img { display: block; width: 100%; border: 0; border-radius: 0; }
    .mask-viewer svg { position: absolute; inset: 0; width: 100%; height: 100%; }
    .mask-path { fill-opacity: .35; stroke: transparent; stroke-width: 2; cursor: pointer; transition: fill-opacity .12s, stroke .12s, stroke-width .12s; }
    .mask-path:hover { fill-opacity: .78; stroke: #ffffff; stroke-width: 4; filter: drop-shadow(0 0 5px rgba(0,0,0,.8)); }
    .mask-preview-panel { margin-top: 16px; padding: 12px; border: 1px solid #e5e7eb; border-radius: 10px; background: #fafafa; }
    .mask-preview-title { margin-bottom: 8px; color: #374151; font-weight: 700; white-space: pre-wrap; }
    .mask-preview-img { display: none; width: min(360px, 100%); border-radius: 8px; border: 1px solid #d1d5db; image-rendering: pixelated; background: #000; }
    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
    .status { color: #2563eb; margin-top: 10px; white-space: pre-wrap; }
    .hint { color: #6b7280; font-size: 14px; }
  </style>
</head>
<body>
  <h1>Semantic-SAM Web UI</h1>
  <p class="hint">默认使用 All (1-6) + points_per_side=64，适合生成较细粒度的 part-level masks。</p>

  <div class="card">
    <h2>Auto Generation</h2>
    <div class="row">
      <form id="autoForm">
        <label>Image</label><input name="image" type="file" accept="image/*" required />
        <label>Granularity</label>
        <select name="level_choice">
          <option>All (1-6)</option><option>Prompt 3</option><option>Prompt 1</option><option>Prompt 2</option>
          <option>Prompt 4</option><option>Prompt 5</option><option>Prompt 6</option>
        </select>
        <label>points_per_side</label><input name="points_per_side" type="number" min="4" max="64" step="4" value="64" />
        <label>pred_iou_thresh</label><input name="pred_iou_thresh" type="number" min="0" max="1" step="0.01" value="0.80" />
        <label>stability_score_thresh</label><input name="stability_score_thresh" type="number" min="0" max="1" step="0.01" value="0.90" />
        <label>postprocess</label>
        <select name="postprocess">
          <option value="small_first_cut">small_first_cut</option>
          <option value="hierarchy_cut">hierarchy_cut</option>
          <option value="none">none</option>
        </select>
        <label>min_mask_area</label><input name="min_mask_area" type="number" min="1" step="1" value="1" />
        <label>min_visible_ratio</label><input name="min_visible_ratio" type="number" min="0" max="1" step="0.01" value="0.01" />
        <label>contain_thresh</label><input name="contain_thresh" type="number" min="0" max="1" step="0.01" value="0.80" />
        <button type="submit">Run Auto</button>
        <div id="autoStatus" class="status"></div>
      </form>
      <div id="autoOutput"></div>
    </div>
  </div>

<script>
async function postForm(form, url, statusEl) {
  statusEl.textContent = 'Running...';
  const res = await fetch(url, { method: 'POST', body: new FormData(form) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || data.error || JSON.stringify(data));
  return data;
}

document.getElementById('autoForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const status = document.getElementById('autoStatus');
  try {
    const data = await postForm(e.target, '/api/auto', status);
    status.textContent = data.info;
    document.getElementById('autoOutput').innerHTML = data.html;
    bindMaskHoverPreview();
  } catch (err) { status.textContent = 'Error: ' + err.message; }
});

function bindMaskHoverPreview() {
  const output = document.getElementById('autoOutput');
  const previewImg = output.querySelector('.mask-preview-img');
  const previewTitle = output.querySelector('.mask-preview-title');
  output.querySelectorAll('.mask-path').forEach((path) => {
    path.addEventListener('mouseenter', () => {
      previewImg.src = path.dataset.mask;
      previewImg.style.display = 'block';
      previewTitle.textContent = path.dataset.title;
    });
  });
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/api/auto")
def api_auto(
    image: UploadFile = File(...),
    level_choice: str = Form("All (1-6)"),
    points_per_side: int = Form(64),
    pred_iou_thresh: float = Form(0.80),
    stability_score_thresh: float = Form(0.90),
    postprocess: str = Form("small_first_cut"),
    min_mask_area: int = Form(1),
    min_visible_ratio: float = Form(0.01),
    contain_thresh: float = Form(0.80),
):
    start = time.time()
    print(f"[Auto] start {level_choice=} {points_per_side=}", flush=True)
    if level_choice == "All (1-6)":
        level = [1, 2, 3, 4, 5, 6]
    else:
        level = [int(level_choice.split(" ")[-1])]
    try:
        with infer_lock:
            pil = read_upload_to_pil(image.file.read())
            image_ori, images = prepare_image_from_pil(pil)
            generator = SemanticSamAutomaticMaskGenerator(
                model,
                points_per_side=int(points_per_side),
                points_per_batch=64,
                pred_iou_thresh=float(pred_iou_thresh),
                stability_score_thresh=float(stability_score_thresh),
                level=level,
            )
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                masks = generator.generate(images)
            raw_count = len(masks)
            if postprocess == "small_first_cut":
                masks = small_first_cut_masks(masks, min_mask_area=min_mask_area, min_visible_ratio=min_visible_ratio)
            elif postprocess == "hierarchy_cut":
                masks = hierarchy_cut_masks(
                    masks,
                    min_mask_area=min_mask_area,
                    min_visible_ratio=min_visible_ratio,
                    contain_thresh=contain_thresh,
                )
            hover_html = render_hover_html(image_ori, masks)
            torch.cuda.empty_cache()
        elapsed = time.time() - start
        info = f"Level={level} | raw_masks={raw_count} | masks={len(masks)} | postprocess={postprocess} | elapsed={elapsed:.2f}s"
        print(f"[Auto] done {info}", flush=True)
        return {"html": hover_html, "info": info}
    except Exception as exc:
        print(f"[Auto] error {repr(exc)}", flush=True)
        return JSONResponse(status_code=500, content={"error": repr(exc)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)

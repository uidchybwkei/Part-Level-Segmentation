import numpy as np


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

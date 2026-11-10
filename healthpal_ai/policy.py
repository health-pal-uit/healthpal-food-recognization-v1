"""Frozen P0/P1/P2 rules. No training or confidence calibration."""
from dataclasses import dataclass
from enum import Enum


class Policy(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str


def box_iou(a, b):
    area = lambda v: max(0, v[2] - v[0]) * max(0, v[3] - v[1])
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / max(1e-12, area(a) + area(b) - intersection)


def decide(original, brightened, policy: Policy, config) -> Decision:
    # Both candidates are already the top detection across ALL native outputs.
    if original is None or original["confidence"] < config["candidate_floor"]:
        return Decision("no_detection", "no_candidate")
    if original["class_id"] not in config["target_class_ids"]:
        return Decision("uncertain", "outside_catalogue")
    if policy != Policy.P0 and original["confidence"] < config["confidence_threshold"]:
        return Decision("uncertain", "below_threshold")
    if policy == Policy.P2:
        if brightened is None or brightened["confidence"] < config["bright_candidate_floor"]:
            return Decision("uncertain", "no_bright_candidate")
        if original["class_id"] != brightened["class_id"]:
            return Decision("uncertain", "view_label_disagreement")
        if box_iou(original["xyxy"], brightened["xyxy"]) < config["stability_iou"]:
            return Decision("uncertain", "view_box_disagreement")
    return Decision("accepted", "accepted")

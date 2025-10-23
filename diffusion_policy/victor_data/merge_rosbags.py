#!/usr/bin/env python3
"""
Merge topics from A into B with B's timeline, per run, with safety + topic filtering.

Changes requested:
1) If B/rosbag_incomplete exists, DO NOT move B/rosbag again. Use rosbag_incomplete as B source.
2) Only import topics from A that are NOT present in B's metadata (name match). If a topic exists
   in B, skip it even if B's topic is empty.

A root: ~/datasets/robotool_hdd
B root: ~/datasets/robotool
A pattern: {YYYYMMDD}_{HHMMSS}_cap{51..102}.*
B pattern: {YYYYMMDD}_{HHMMSS}_replayof_cap{51..102}.*
"""

import argparse
import bisect
import hashlib
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

import yaml

# ROS 2
try:
    import rosbag2_py
except ImportError as e:
    print("ERROR: rosbag2_py not found. Source your ROS 2 Humble setup.", file=sys.stderr)
    raise

COMMAND_TOPIC = "/right_arm_impedance_controller/commands"

CAP_RE_A = re.compile(r"(?P<date>\d{8})_(?P<hms>\d{6})_cap(?P<cap>\d{2,3})")
CAP_RE_B = re.compile(r"(?P<date>\d{8})_(?P<hms>\d{6})_replayof_cap(?P<cap>\d{2,3})")

@dataclass(frozen=True)
class RunKey:
    date: str
    hms: str
    cap: int

def parse_a_key(p: Path) -> Optional[RunKey]:
    m = CAP_RE_A.search(p.name)
    return RunKey(m.group("date"), m.group("hms"), int(m.group("cap"))) if m else None

def parse_b_key(p: Path) -> Optional[RunKey]:
    m = CAP_RE_B.search(p.name)
    return RunKey(m.group("date"), m.group("hms"), int(m.group("cap"))) if m else None

def find_candidate_runs(root: Path, kind: str) -> Dict[RunKey, Path]:
    out: Dict[RunKey, Path] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        key = parse_a_key(child) if kind == "A" else parse_b_key(child)
        if key:
            out[key] = child
    return out

def find_rosbag_dir(run_dir: Path) -> Optional[Path]:
    # Prefer run_dir/rosbag with metadata.yaml
    rb = run_dir / "rosbag"
    if (rb / "metadata.yaml").exists():
        return rb
    # Otherwise, look shallow/deep
    for child in run_dir.iterdir():
        if child.is_dir() and (child / "metadata.yaml").exists():
            return child
    for meta in run_dir.rglob("metadata.yaml"):
        return meta.parent
    return None

def read_storage_id(bag_dir: Path) -> str:
    meta = bag_dir / "metadata.yaml"
    with open(meta, "r") as f:
        m = yaml.safe_load(f)
    sid = (m.get("storage_identifier")
           or m.get("rosbag2_bagfile_information", {}).get("storage_identifier"))
    return sid if sid in ("sqlite3", "mcap") else "sqlite3"

def open_reader(bag_dir: Path, storage_id: str) -> rosbag2_py.SequentialReader:
    r = rosbag2_py.SequentialReader()
    r.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", "")
    )
    return r

def get_topics(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    return {t.name: t.type for t in reader.get_all_topics_and_types()}

def create_writer(out_dir: Path, topics_and_types: Dict[str, str], storage_id: str) -> rosbag2_py.SequentialWriter:
    # If a previous attempt left anything in the output folder, purge it.
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Important: rosbag2 wants to create `out_dir` itself.
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    w = rosbag2_py.SequentialWriter()
    w.open(
        rosbag2_py.StorageOptions(uri=str(out_dir), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", "")
    )
    for name, typ in topics_and_types.items():
        md = rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr")
        w.create_topic(md)
    return w

def hash_payload(data: bytes) -> bytes:
    return hashlib.sha1(data).digest()

@dataclass
class AnchorIndex:
    a_times: List[int]
    b_times: List[int]
    def shift_ts(self, ts_a: int) -> Optional[int]:
        if not self.a_times:
            return None
        i = bisect.bisect_left(self.a_times, ts_a)
        if i == 0:
            i0 = 0
        elif i >= len(self.a_times):
            i0 = len(self.a_times) - 1
        else:
            left = i - 1
            right = i
            i0 = left if abs(self.a_times[left] - ts_a) <= abs(self.a_times[right] - ts_a) else right
        dt = self.b_times[i0] - self.a_times[i0]
        return ts_a + dt

def build_anchor_index(a_bag: Path, b_bag: Path, storage_a: str, storage_b: str, min_matches: int = 5) -> AnchorIndex:
    a_reader = open_reader(a_bag, storage_a)
    a_hash_to_times: Dict[bytes, List[int]] = {}
    while a_reader.has_next():
        topic, data, t = a_reader.read_next()
        if topic == COMMAND_TOPIC:
            h = hash_payload(data)
            a_hash_to_times.setdefault(h, []).append(t)

    b_reader = open_reader(b_bag, storage_b)
    pairs: List[Tuple[int, int]] = []
    while b_reader.has_next():
        topic, data, t = b_reader.read_next()
        if topic == COMMAND_TOPIC:
            h = hash_payload(data)
            if h in a_hash_to_times:
                a_list = a_hash_to_times[h]
                j = bisect.bisect_left(a_list, t)
                if j == 0:
                    a_match = a_list[0]
                elif j >= len(a_list):
                    a_match = a_list[-1]
                else:
                    left = a_list[j-1]; right = a_list[j]
                    a_match = left if abs(left - t) <= abs(right - t) else right
                pairs.append((a_match, t))

    pairs.sort(key=lambda x: x[0])
    dedup: Dict[int, int] = {}
    for a_t, b_t in pairs:
        if a_t not in dedup or abs(dedup[a_t] - a_t) > abs(b_t - a_t):
            dedup[a_t] = b_t
    a_times = sorted(dedup.keys())
    b_times = [dedup[a] for a in a_times]
    if len(a_times) < min_matches:
        print(f"  WARNING: Only {len(a_times)} anchor matches found; timestamp shifting may be coarse.", file=sys.stderr)
    return AnchorIndex(a_times=a_times, b_times=b_times)

def copy_b_to_writer(b_bag: Path, storage_b: str, writer: rosbag2_py.SequentialWriter):
    r = open_reader(b_bag, storage_b)
    while r.has_next():
        topic, data, t = r.read_next()
        writer.write(topic, data, t)
# `
# def write_selected_from_a(a_bag: Path,
#                           storage_a: str,
#                           writer: rosbag2_py.SequentialWriter,
#                           anchor: AnchorIndex,
#                           topics_from_a: Set[str]) -> int:
#     r = open_reader(a_bag, storage_a)
#     n_written = 0
#     while r.has_next():
#         topic, data, t = r.read_next()
#         if topic not in topics_from_a:
#             continue
#         if topic == COMMAND_TOPIC:
#             continue
#         t_new = anchor.shift_ts(t) if anchor else None
#         writer.write(topic, data, int(t if t_new is None else t_new))
#         n_written += 1
#     return n_written
# 
def write_selected_from_a_tfaware(
    a_bag: Path,
    storage_a: str,
    writer: rosbag2_py.SequentialWriter,
    anchor: AnchorIndex,
    topics_from_a: set[str],
    b_tf_static_children: set[str],
    b_tf_dynamic_pairs: set[tuple[str, str]],
) -> int:
    """
    Write messages from A for the selected topics.
    Special handling:
      - If topic == /tf_static: only write transforms whose child_frame_id not in B's static set.
      - If topic == /tf: only write transforms whose (parent, child) pair not in B's dynamic set.
    """
    r = open_reader(a_bag, storage_a)
    n_written = 0

    while r.has_next():
        topic, data, t = r.read_next()

        # Skip topics not explicitly selected from A (including TF topics for frame-level merge)
        if topic not in topics_from_a:
            continue

        # Never import the command topic from A
        if topic == COMMAND_TOPIC:
            continue

        # Default: timestamp shift
        t_new = anchor.shift_ts(t) if anchor else None
        t_out = int(t if t_new is None else t_new)

        # TF filtering
        if topic in ("/tf", "/tf_static"):
            try:
                msg_in = deserialize_message(data, TFMessage)
            except Exception:
                # If we cannot deserialize, skip safely
                continue

            filtered = []
            if topic == "/tf_static":
                for ts in msg_in.transforms:
                    if ts.child_frame_id not in b_tf_static_children:
                        filtered.append(ts)
                if not filtered:
                    continue  # nothing new to add
            else:  # "/tf"
                for ts in msg_in.transforms:
                    key = (ts.header.frame_id, ts.child_frame_id)
                    if key not in b_tf_dynamic_pairs:
                        filtered.append(ts)
                if not filtered:
                    continue  # nothing new to add

            # Build an outgoing TFMessage with only the filtered transforms.
            out_msg = TFMessage(transforms=filtered)

            # Serialize the outgoing message back to CDR
            # Note: rosbag2_py writer.write() expects serialized bytes;
            # however, Humble's writer can accept raw bytes from reader.
            # For safety, we'll re-serialize the message.
            from rclpy.serialization import serialize_message
            out_bytes = serialize_message(out_msg)

            writer.write(topic, out_bytes, t_out)
            n_written += 1
            continue

        # Non-TF topics: normal pass-through for A-selected topics
        writer.write(topic, data, t_out)
        n_written += 1

    return n_written

def union_for_writer(b_topics: Dict[str, str], a_topics: Dict[str, str], a_topics_allowed: Set[str]) -> Dict[str, str]:
    out = dict(b_topics)
    for n in a_topics_allowed:
        if n == COMMAND_TOPIC:
            continue
        out.setdefault(n, a_topics[n])
    return out


def collect_tf_presence(bag_dir: Path, storage_id: str) -> tuple[set[str], set[tuple[str, str]]]:
    """
    Return:
      static_children: set[str] of child_frame_id present in /tf_static
      dynamic_pairs: set[(parent, child)] present in /tf
    """
    static_children: set[str] = set()
    dynamic_pairs: set[tuple[str, str]] = set()

    if not bag_dir or not (bag_dir / "metadata.yaml").exists():
        return static_children, dynamic_pairs

    r = open_reader(bag_dir, storage_id)
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic not in ("/tf", "/tf_static"):
            continue
        try:
            msg = deserialize_message(data, TFMessage)
        except Exception:
            continue  # skip any bad record
        if topic == "/tf_static":
            for ts in msg.transforms:
                static_children.add(ts.child_frame_id)
        else:  # "/tf"
            for ts in msg.transforms:
                dynamic_pairs.add((ts.header.frame_id, ts.child_frame_id))
    return static_children, dynamic_pairs


def main():
    parser = argparse.ArgumentParser(description="Merge A→B topics onto B's timeline, safely.")
    parser.add_argument("--a-root", default="/home/houhd/datasets/robotool_hdd", type=str)
    parser.add_argument("--b-root", default="/home/houhd/datasets/robotool", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    a_root = Path(os.path.expanduser(args.a_root))
    b_root = Path(os.path.expanduser(args.b_root))
    print(a_root, b_root)

    runs_a = find_candidate_runs(a_root, "A")  # Dict[RunKey, Path]
    runs_b = find_candidate_runs(b_root, "B")  # Dict[RunKey, Path]

    # ----- Pair by CAP only -----
    from collections import defaultdict

    def _sort_key(k: RunKey) -> tuple[str, str]:
        return (k.date, k.hms)

    a_by_cap: dict[int, list[tuple[RunKey, Path]]] = defaultdict(list)
    for k, p in runs_a.items():
        a_by_cap[k.cap].append((k, p))

    b_by_cap: dict[int, list[tuple[RunKey, Path]]] = defaultdict(list)
    for k, p in runs_b.items():
        b_by_cap[k.cap].append((k, p))

    matched_caps = sorted(set(a_by_cap.keys()) & set(b_by_cap.keys()))
    if not matched_caps:
        print("No matching A/B runs found.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(matched_caps)} matched caps.")
    for cap in matched_caps:
        # pick most recent A and most recent B for this cap
        kA, runA = sorted(a_by_cap[cap], key=lambda kp: _sort_key(kp[0]))[-1]
        kB, runB = sorted(b_by_cap[cap], key=lambda kp: _sort_key(kp[0]))[-1]

        print(f"\n=== cap{cap}: A={runA.name}  B={runB.name} ===")

        a_bag = find_rosbag_dir(runA)
        if not a_bag:
            print(f"  SKIP: No A rosbag under {runA}", file=sys.stderr)
            continue

        out_dir = runB / "rosbag"
        incomplete_dir = runB / "rosbag_incomplete"

        # Discover any existing B bag (for preview/anchors/topic list)
        discovered_b = find_rosbag_dir(runB)
        has_out_dir = (out_dir / "metadata.yaml").exists()
        has_incomplete = (incomplete_dir / "metadata.yaml").exists()
        has_discovered_elsewhere = (
            discovered_b is not None
            and discovered_b != out_dir
            and (discovered_b / "metadata.yaml").exists()
        )

        # Decide the *effective* B source to READ FROM in this run,
        # and what we will MOVE (or would move in dry-run).
        b_source_dir: Optional[Path] = None          # actual path we'll read from NOW
        planned_b_source_after_move: Optional[Path] = None  # only for printing in dry-run

        if has_incomplete:
            # Resume case: use incomplete directly; do not move anything.
            print("  Detected existing rosbag_incomplete (resuming). Will NOT move B/rosbag.")
            b_source_dir = incomplete_dir
        else:
            # No rosbag_incomplete yet
            if has_out_dir:
                if args.dry_run:
                    print(f"  Would move {out_dir} -> {incomplete_dir}")
                    b_source_dir = out_dir  # still exists now; safe to open for preview
                    planned_b_source_after_move = incomplete_dir
                else:
                    print(f"  Moving {out_dir} -> {incomplete_dir}")
                    if incomplete_dir.exists():
                        shutil.rmtree(incomplete_dir)
                    shutil.move(str(out_dir), str(incomplete_dir))
                    b_source_dir = incomplete_dir
            elif has_discovered_elsewhere:
                if args.dry_run:
                    print(f"  Would move {discovered_b} -> {incomplete_dir}")
                    b_source_dir = discovered_b  # still exists now
                    planned_b_source_after_move = incomplete_dir
                else:
                    print(f"  Moving {discovered_b} -> {incomplete_dir}")
                    if incomplete_dir.exists():
                        shutil.rmtree(incomplete_dir)
                    shutil.move(str(discovered_b), str(incomplete_dir))
                    b_source_dir = incomplete_dir
            else:
                print("  No existing B rosbag found; will create from scratch.")
                b_source_dir = None

        storage_a = read_storage_id(a_bag) if (a_bag / "metadata.yaml").exists() else "sqlite3"
        # For storage_b: if we actually have a readable source, detect it; else fallback to A's storage
        storage_b = (
            read_storage_id(b_source_dir)
            if (b_source_dir and (b_source_dir / "metadata.yaml").exists())
            else storage_a
        )

        if args.dry_run:
            print(f"  A bag: {a_bag} [{storage_a}]")
            if b_source_dir:
                extra = f" (planned B source after move: {planned_b_source_after_move})" if planned_b_source_after_move else ""
                print(f"  B source (for preview): {b_source_dir} [{storage_b}]{extra}")
            else:
                print("  B source (for preview): (none)")

        # Build topic sets
        b_types: Dict[str, str] = {}
        a_types: Dict[str, str] = get_topics(open_reader(a_bag, storage_a))
        if b_source_dir and (b_source_dir / "metadata.yaml").exists():
            b_types = get_topics(open_reader(b_source_dir, storage_b))

        # Only import A topics NOT present in B (by name). Always exclude COMMAND_TOPIC from A.
        a_only_topics: Set[str] = {t for t in a_types.keys() if t not in b_types}
        a_only_topics.discard(COMMAND_TOPIC)

        # Always consider TF topics for frame-level merge (even if present in B)
        force_tf_topics: Set[str] = set()
        if "/tf" in a_types:
            force_tf_topics.add("/tf")
        if "/tf_static" in a_types:
            force_tf_topics.add("/tf_static")

        # Collect TF presence from B for filtering
        b_tf_static_children: set[str] = set()
        b_tf_dynamic_pairs: set[tuple[str, str]] = set()
        if b_source_dir and (b_source_dir / "metadata.yaml").exists():
            b_tf_static_children, b_tf_dynamic_pairs = collect_tf_presence(b_source_dir, storage_b)

        # Only import A topics NOT present in B (by name). Always exclude COMMAND_TOPIC from A.
        a_only_topics: Set[str] = {t for t in a_types.keys() if t not in b_types}
        a_only_topics.discard(COMMAND_TOPIC)

        # Check if we have anything to process (either A-only topics or TF topics for frame-level merge)
        has_work_to_do = bool(a_only_topics) or bool(force_tf_topics)
        
        if not has_work_to_do:
            print("  Nothing to import from A (all topics already exist in B and no TF frames to merge).")
            # If there is no out_dir but there is b_source_dir, mirror B to out_dir so B/rosbag exists.
            if b_source_dir and not args.dry_run:
                writer = create_writer(out_dir, b_types, storage_b)
                print("  Copying B messages to new rosbag (no A topics needed)…")
                copy_b_to_writer(b_source_dir, storage_b, writer)
            continue

        if args.dry_run:
            print(f"  Will import {len(a_only_topics)} A-only topics:")
            for t in sorted(a_only_topics):
                print(f"    {t}")
            if force_tf_topics:
                print(f"  Will also process {len(force_tf_topics)} TF topics for frame-level merge:")
                for t in sorted(force_tf_topics):
                    print(f"    {t}")
            # Preview anchors only if we have an actual readable B source
            if b_source_dir and (b_source_dir / "metadata.yaml").exists():
                anchor = build_anchor_index(a_bag, b_source_dir, storage_a, storage_b)
                print(f"  Anchor matches: {len(anchor.a_times)}")
            else:
                print("  Anchor preview skipped (no readable B source).")
            if "/tf" in a_types or "/tf_static" in a_types:
                a_tf_static_children, a_tf_dynamic_pairs = collect_tf_presence(a_bag, storage_a)
                if b_source_dir and (b_source_dir / "metadata.yaml").exists():
                    b_tf_static_children, b_tf_dynamic_pairs = collect_tf_presence(b_source_dir, storage_b)
                else:
                    b_tf_static_children, b_tf_dynamic_pairs = set(), set()

                missing_static = sorted(a_tf_static_children - b_tf_static_children)
                missing_dynamic = sorted(a_tf_dynamic_pairs - b_tf_dynamic_pairs)

                print("  TF dry-run preview:")
                if "/tf_static" in a_types:
                    print(f"    /tf_static: would add {len(missing_static)} child_frame_id(s)")
                    # list a few for visibility
                    for name in missing_static[:20]:
                        print(f"      - {name}")
                    if len(missing_static) > 20:
                        print(f"      ... and {len(missing_static) - 20} more")
                else:
                    print("    /tf_static: not present in A (nothing to add)")

                if "/tf" in a_types:
                    print(f"    /tf: would add {len(missing_dynamic)} (parent, child) pair(s)")
                    for pair in missing_dynamic[:20]:
                        print(f"      - {pair[0]} -> {pair[1]}")
                    if len(missing_dynamic) > 20:
                        print(f"      ... and {len(missing_dynamic) - 20} more")
                else:
                    print("    /tf: not present in A (nothing to add)")
            continue

        # Prepare writer topics: B topics + (A-only topics) + (force TF topics)
        topics_for_writer = set(a_only_topics) | force_tf_topics
        topics_union = union_for_writer(b_types, a_types, topics_for_writer)
        writer_storage = storage_b if b_source_dir else storage_a
        writer = create_writer(out_dir, topics_union, writer_storage)

        # Copy B messages first (if any)
        if b_source_dir and (b_source_dir / "metadata.yaml").exists():
            print("  Copying B messages…")
            copy_b_to_writer(b_source_dir, storage_b, writer)

        # Build anchors and import A-only topics aligned to B-time
        print("  Building A↔B command anchors…")
        anchor = (
            build_anchor_index(a_bag, b_source_dir, storage_a, storage_b)
            if (b_source_dir and (b_source_dir / "metadata.yaml").exists())
            else AnchorIndex([], [])
        )

        print("  Importing A-only topics with aligned timestamps…")
        # n = write_selected_from_a(a_bag, storage_a, writer, anchor, a_only_topics)
        # Topics we will attempt to pull from A:
        topics_from_a_effective = set(a_only_topics) | force_tf_topics

        n = write_selected_from_a_tfaware(
            a_bag,
            storage_a,
            writer,
            anchor,
            topics_from_a_effective,
            b_tf_static_children,
            b_tf_dynamic_pairs,
        )
        print(f"  Wrote {n} messages from A into {out_dir}")

    print("\nDone.")

if __name__ == "__main__":
    main()
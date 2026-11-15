import json
from pathlib import Path


def merge_batches(
    batches_dir: Path | str = "data/gd_batches",
    output_file: Path | str = "data/golden_dataset.json",
) -> list[dict]:
    """Merge all domain_*.json batch files into data/golden_dataset.json sorted by ID."""
    b_dir = Path(batches_dir)
    out_path = Path(output_file)

    merged: dict[str, dict] = {}
    if b_dir.exists():
        for batch_file in sorted(b_dir.glob("*.json")):
            raw_text = batch_file.read_text(encoding="utf-8-sig")
            entries = json.loads(raw_text)
            # Re-save the batch file as clean UTF-8 without BOM
            batch_file.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            for item in entries:
                entry_id = item["id"]
                if entry_id in merged:
                    raise ValueError(f"Duplicate Golden Dataset ID '{entry_id}' found in {batch_file.name}")
                merged[entry_id] = item

    sorted_entries = [merged[k] for k in sorted(merged.keys())]
    out_path.write_text(json.dumps(sorted_entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Merged {len(sorted_entries)} entries into {out_path} (clean UTF-8 no BOM)")
    return sorted_entries


if __name__ == "__main__":
    merge_batches()

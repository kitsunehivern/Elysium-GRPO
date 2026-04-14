from pathlib import Path
import tarfile
import sys

archive = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else archive.parent / archive.stem.replace(".tar", "")

output_dir.mkdir(parents=True, exist_ok=True)

print(f"Extracting: {archive}")
print(f"To: {output_dir}")

with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(output_dir)

print("Done. Archive kept:", archive)

"""
python scripts/extract_elysium.py \
/raid/hvtham/dcmquan/Elysium/datasets/train/ElysiumTrack-1M/packaged_results/group_aa.tar.gz \
/raid/hvtham/dcmquan/Elysium/datasets/train/ElysiumTrack-1M/packaged_results
"""
import argparse
from pathlib import Path
import mne
from mne.datasets import eegbci

p = argparse.ArgumentParser()
p.add_argument("--subjects", nargs="+", type=int, default=[1])
p.add_argument("--runs", nargs="+", type=int, default=[1, 2])
p.add_argument("--data-dir", default="data/raw")
args = p.parse_args()

out = Path(args.data_dir)
out.mkdir(parents=True, exist_ok=True)
for subject in args.subjects:
    paths = eegbci.load_data(subject, args.runs, path=str(out), update_path=True)
    for src in paths:
        src = Path(src)
        dst = out / src.name
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        print(f"Available: {dst}")
print("Download complete.")

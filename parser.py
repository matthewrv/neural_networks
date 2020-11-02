"""Decompressor for MNIST database"""
from pathlib import Path
import gzip

resources_path = Path("../resources")
for file in resources_path.glob("*.gz"):
    data = gzip.decompress(file.read_bytes())
    new_file = resources_path / file.stem
    new_file.write_bytes(data)

#!/usr/bin/env python3
"""
Script to collect all .gif files from subdirectories of out/ and move them to src/zip
"""
import os
import shutil
from pathlib import Path

# Define paths
out_dir = Path("camera/output")
zip_dest = Path("src/zip")

# Create destination directory if it doesn't exist
zip_dest.mkdir(parents=True, exist_ok=True)

# Find and move all .gif files
if out_dir.exists():
    gif_files = list(out_dir.rglob("*.gif"))
    
    if gif_files:
        print(f"Found {len(gif_files)} .gif file(s)")
        for gif_file in gif_files:
            # Preserve uniqueness by including parent directory name
            parent_name = gif_file.parent.name
            new_name = f"{parent_name}_{gif_file.name}"
            dest_path = zip_dest / new_name
            print(f"Moving {gif_file} -> {dest_path}")
            shutil.move(str(gif_file), str(dest_path))
        print(f"\nAll .gif files moved to {zip_dest}")
    else:
        print(f"No .gif files found in {out_dir}")
else:
    print(f"Error: Directory '{out_dir}' does not exist")

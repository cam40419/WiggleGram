"""Test script for manual alignment feature."""
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from pipeline import run_pipeline


def main():
    """Test manual alignment on a specific image."""
    # Enable manual alignment mode
    cfg.set_runtime("manual_alignment", True)
    
    # Prompt for input file
    print("=" * 60)
    print("Manual Alignment Test")
    print("=" * 60)
    
    input_path = input("\nEnter path to raw image (or press Enter for default): ").strip()
    
    if not input_path:
        # Use default path from config
        input_dir = cfg.INPUT_DIR
        if not input_dir.exists():
            print(f"Error: Default input directory not found: {input_dir}")
            return
        
        # Find most recent image
        images = sorted(input_dir.glob("photo_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not images:
            print(f"Error: No images found in {input_dir}")
            return
        
        input_path = str(images[0])
        print(f"Using most recent image: {input_path}")
    
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return
    
    # Set output directory
    output_dir = cfg.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract timestamp from filename
    timestamp = input_file.stem.replace("photo_", "")
    
    print(f"\nProcessing: {input_file.name}")
    print(f"Output will be saved to: {output_dir / f'{timestamp}.gif'}")
    print("\nManual alignment GUI will open. Click the same point on each camera view.")
    print("Press Enter to continue...")
    input()
    
    try:
        run_pipeline(str(input_file), str(output_dir), timestamp)
        print(f"\n✓ Success! GIF saved to: {output_dir / f'{timestamp}.gif'}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

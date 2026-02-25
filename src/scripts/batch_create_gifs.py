#!/usr/bin/env python3
"""
Batch process JPG images to create wigglegrams.

This script processes all JPG images in a directory, splitting each 2x2 grid
image into 4 pieces and running the full wigglegram pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import config as cfg
from pipeline import (
    split_grid,
    apply_rotation_calibration,
    apply_white_balance,
    apply_color_correction,
    pick_anchors_template,
    wigglegram_anchors,
)
from manual_align import get_manual_anchors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def find_jpg_images(input_dir: Path) -> List[Path]:
    """Find all JPG images in a directory."""
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return []
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return []
    
    jpg_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
        jpg_files.extend(input_dir.glob(ext))
    
    # Sort by filename for consistent ordering
    jpg_files = sorted(jpg_files)
    
    logger.info(f"Found {len(jpg_files)} JPG images in {input_dir}")
    return jpg_files


def process_image(image_path: Path, output_dir: Path, manual_mode: bool = False):
    """Process a single 2x2 grid image into a wigglegram GIF."""
    logger.info(f"Processing: {image_path.name}")
    
    try:
        # Load the raw image
        raw = Image.open(image_path).convert("RGB")
        logger.debug(f"Loaded image: {raw.size}")
        
        # Split into 2x2 grid (4 pieces)
        pieces = split_grid(raw, 2, 2)
        logger.debug(f"Split into {len(pieces)} pieces")
        
        # Rotate each piece 90 degrees
        pieces = [p.transpose(Image.Transpose.ROTATE_90) for p in pieces]
        
        # Apply calibrations if available
        pieces = apply_rotation_calibration(pieces)
        pieces = apply_white_balance(pieces)
        # Optionally enable color correction
        pieces = apply_color_correction(pieces)
        
        # Ensure all pieces are the same size
        base_w, base_h = pieces[0].size
        pieces = [img.resize((base_w, base_h), Image.Resampling.BILINEAR) 
                  if img.size != (base_w, base_h) else img 
                  for img in pieces]
        
        # Pick anchor points for alignment
        target_xy = (base_w // 2, base_h // 2)
        
        if manual_mode:
            logger.info("Manual alignment mode - GUI will open")
            anchors = get_manual_anchors(pieces)
            if anchors is None:
                logger.warning("Manual alignment cancelled - falling back to automatic")
                anchors = pick_anchors_template(pieces, target_xy=target_xy, ref_idx=1)
        else:
            anchors = pick_anchors_template(pieces, target_xy=target_xy, ref_idx=1)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the original filename (without extension) for the GIF
        gif_filename = image_path.stem
        
        # Generate the wigglegram
        wigglegram_anchors(
            pieces,
            anchors=anchors,
            save_dir=str(output_dir),
            filename=gif_filename,
            target_xy=target_xy,
            axis="x",
            duration_ms=100,
            crop_common=True,
        )
        
        logger.info(f"✓ Created GIF: {output_dir / f'{gif_filename}.gif'}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to process {image_path.name}: {e}", exc_info=True)
        return False


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(
        description="Batch process JPG images into wigglegram GIFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all JPG images in a directory
  python batch_create_gifs.py /path/to/input /path/to/output
  
  # Use default output directory
  python batch_create_gifs.py /path/to/input
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing JPG images (2x2 grid format)"
    )
    
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Output directory for GIF files (default: input_dir/output)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )
    
    parser.add_argument(
        "-m", "--manual",
        action="store_true",
        help="Enable manual anchor point selection (GUI for each image)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up paths
    input_dir = args.input_dir.resolve()
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = input_dir / "output"
    
    logger.info("=" * 60)
    logger.info("Wigglegram Batch Processor")
    logger.info("=" * 60)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    alignment_mode = "Manual" if args.manual else "Automatic"
    logger.info(f"Alignment mode:   {alignment_mode}")
    logger.info("=" * 60)
    
    # Find all JPG images
    jpg_files = find_jpg_images(input_dir)
    
    if not jpg_files:
        logger.warning("No JPG images found to process")
        return 1
    
    logger.info(f"Processing {len(jpg_files)} images...")
    logger.info("")
    
    # Process each image
    success_count = 0
    fail_count = 0
    
    for i, image_path in enumerate(jpg_files, 1):
        logger.info(f"[{i}/{len(jpg_files)}] {image_path.name}")
        if process_image(image_path, output_dir, manual_mode=args.manual):
            success_count += 1
        else:
            fail_count += 1
        logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Batch processing complete!")
    logger.info(f"Success: {success_count} | Failed: {fail_count} | Total: {len(jpg_files)}")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("=" * 60)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

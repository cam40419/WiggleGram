# Manual Anchor Point Selection

This feature allows you to manually select alignment points for each camera view instead of relying on automatic template matching. This can produce better results when:

- The subject moves between frames
- There's motion blur in the images
- The automatic template matching fails to find a good match
- You want precise control over the stabilization point

## How to Use

### Main Application (app.py)

When you start the main application, you'll be prompted:

```
Enable manual anchor point selection?

Yes = Manually click alignment points for each photo
No = Automatic template-based alignment
```

- Click **Yes** to enable manual mode
- Click **No** for automatic mode (default behavior)

### Batch Processing Script

Use the `-m` or `--manual` flag to enable manual alignment:

```bash
# Automatic mode (default)
python src/scripts/batch_create_gifs.py /path/to/input

# Manual mode
python src/scripts/batch_create_gifs.py -m /path/to/input
```

**Note:** Manual mode for batch processing will open the GUI for every image, which can be time-consuming for large batches.

### Test Script

A dedicated test script is available for testing manual alignment on single images:

```bash
python src/scripts/test_manual_align.py
```

This will prompt you for an input file or use the most recent image from the default input directory.

## Manual Alignment GUI

When manual mode is enabled, a window will open showing each camera view one at a time.

### Instructions:

1. **Select the same point on each image**
   - Click on a distinctive feature (e.g., center of subject's nose, eye, etc.)
   - The point should be clearly visible in all camera views
   - Choose a point that represents where you want the wiggle effect centered

2. **Navigation**
   - Click **Next →** or press **Right Arrow** to move to the next camera
   - Click **← Previous** or press **Left Arrow** to go back
   - Current image number is shown in the center (e.g., "Image 2/4")

3. **Preview Thumbnails**
   - All camera views are shown as thumbnails at the bottom
   - ✓ = Point selected for this camera
   - ✗ = Point not yet selected
   - Blue border = Currently viewing

4. **Generate**
   - The **✓ Generate** button activates when all points are selected
   - Click it or press **Enter** to confirm and create the wigglegram
   - If you close the window without generating, it will fall back to automatic alignment

### Visual Feedback:

- A **red crosshair** marks your selected point on each image
- The crosshair includes a circle at the center for precision
- You can change your selection by clicking again

## Configuration

### Runtime Configuration

The manual alignment mode is controlled by the runtime config:

```python
import config as cfg

# Enable manual mode
cfg.set_runtime("manual_alignment", True)

# Disable manual mode (use automatic)
cfg.set_runtime("manual_alignment", False)

# Check current setting
manual_mode = cfg.get_runtime("manual_alignment", False)
```

### Future Enhancement

This will be moved to a persistent config setting in a future update, allowing you to set your preference without being prompted each time.

## Technical Details

### Implementation

- **Module:** `src/manual_align.py`
- **Main Function:** `get_manual_anchors(images: List[Image.Image])`
- **Returns:** List of (x, y) tuples for each camera view, or None if cancelled

### Integration Points

Manual alignment is integrated into the pipeline at the anchor detection stage:

1. Images are loaded and preprocessed (rotation, calibration, white balance)
2. **Anchor point selection** (manual OR automatic)
3. Frame alignment using anchors
4. Common area cropping
5. GIF generation

### Fallback Behavior

If manual alignment is enabled but:
- The GUI is cancelled/closed
- An error occurs
- No points are selected

The system will automatically fall back to template-based alignment and log a warning.

## Tips for Best Results

1. **Choose a stable feature**
   - Pick something that appears in all camera views
   - Avoid areas near the edge that might be cropped out
   - Good examples: bridge of nose, center of an eye, distinctive facial feature

2. **Be consistent**
   - Click the exact same physical point on each view
   - Zoom in mentally on the feature you're targeting
   - Take your time for precision

3. **Center of subject works best**
   - The wiggle effect is centered on your selected point
   - Choose a point that should remain stable in the final GIF
   - Avoid extremities (hands, feet) unless they're the focus

4. **Test with one image first**
   - Use `test_manual_align.py` to practice
   - Compare results with automatic alignment
   - Determine which works better for your use case

## Comparison with Automatic Alignment

| Feature | Automatic | Manual |
|---------|-----------|--------|
| Speed | Fast | Slower (requires user input) |
| Precision | Good for static scenes | Better for complex scenes |
| Consistency | Highly consistent | Depends on user accuracy |
| Batch Processing | Ideal | Time-consuming |
| Motion Blur | May struggle | User can compensate |
| Subject Movement | May misalign | User selects stable point |

## Troubleshooting

**GUI doesn't open:**
- Check that tkinter is installed: `python -c "import tkinter"`
- Ensure you're not running in a headless environment

**Points not registering:**
- Make sure you're clicking inside the image bounds
- The crosshair should appear immediately after clicking

**Generate button stays disabled:**
- Ensure you've selected a point on ALL camera views
- Check the thumbnail checkmarks (should all show ✓)

**Window too small/large:**
- The window auto-scales to fit your screen
- Images are scaled proportionally to fit the display
- Original coordinates are preserved regardless of display size

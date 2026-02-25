# Bug Fix: Manual Alignment GUI Not Appearing

## Problem
Manual alignment mode was enabled but the GUI never appeared. The pipeline would jump straight to automatic template matching and return to the viewfinder.

## Root Cause
The manual alignment GUI (`get_manual_anchors()`) was being called from a **background thread** in `app.py`. Tkinter GUIs must run on the **main thread**, so the GUI silently failed to open.

```python
# BEFORE (broken):
def process_thread():
    run_pipeline(...)  # This calls get_manual_anchors() from background thread
    
threading.Thread(target=process_thread).start()  # Wrong thread!
```

## Solution
The anchor point selection now happens on the **main thread** before starting the background processing thread:

```python
# AFTER (fixed):
def _start_processing(self, input_file, output_dir):
    # Get anchors on MAIN thread if manual mode enabled
    if cfg.get_runtime("manual_alignment", False):
        selected_anchors = get_manual_anchors(pieces)  # Main thread - works!
    
    # Then pass anchors to background thread
    def process_thread():
        run_pipeline(..., anchors=selected_anchors)
    
    threading.Thread(target=process_thread).start()
```

## Changes Made

### 1. `src/pipeline.py`
- Added optional `anchors` parameter to `run_pipeline()`
- Pipeline now accepts pre-selected anchors instead of always calling GUI
- Improved logging to show when manual vs automatic alignment is used

### 2. `src/app.py`
- Manual alignment GUI now runs on main thread in `_start_processing()`
- Images are preprocessed (split, rotate, calibrate) before showing GUI
- Selected anchors are passed to pipeline in background thread
- Better startup logging to show if manual mode is enabled
- Added environment variable support: `WIGGLEGRAM_MANUAL_ALIGN=true`
- Improved dialog visibility with `-topmost` attribute

### 3. New Files
- `start_manual_mode.sh` - Helper script to launch with manual mode
- `README_MANUAL_MODE.md` - Quick start guide for manual mode
- `BUGFIX_MANUAL_ALIGNMENT.md` - This document

### 4. Enhanced Logging
Manual mode now produces clear log messages:
```
============================================================
MANUAL ALIGNMENT MODE ENABLED
============================================================
Starting app with alignment mode: MANUAL
Manual alignment enabled - showing anchor selection GUI
Manual anchors selected: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
```

## How to Enable Manual Mode Now

### Method 1: Dialog (Click "Yes")
```bash
python3 src/app.py
# Click "Yes" when prompted
```

### Method 2: Environment Variable (Recommended)
```bash
export WIGGLEGRAM_MANUAL_ALIGN=true
python3 src/app.py
```

### Method 3: Helper Script
```bash
chmod +x start_manual_mode.sh
./start_manual_mode.sh
```

## Verification Steps

1. **Check startup logs** for this message:
   ```
   ============================================================
   MANUAL ALIGNMENT MODE ENABLED
   ============================================================
   ```

2. **Take a photo** - you should see:
   ```
   Manual alignment enabled - showing anchor selection GUI
   ```

3. **GUI should appear** showing first camera view with instructions

4. **Click anchor points** on all 4 views

5. **Check logs** for:
   ```
   Manual anchors selected: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
   Using pre-selected anchor points
   ```

## Technical Details

### Threading Model
- **Main Thread**: UI, viewfinder updates, manual alignment GUI
- **Background Thread 1**: Camera capture (`do_capture`)
- **Background Thread 2**: Pipeline processing (GIF generation)

### Why This Matters
Tkinter's event loop and widget creation must happen on the main thread. Attempting to create widgets from background threads causes:
- Silent failures (no error, no GUI)
- Crashes in some Python/Tkinter versions
- Undefined behavior

### Image Preprocessing
To show the manual alignment GUI before the background thread starts, we preprocess the image on the main thread:
1. Load raw 2x2 grid image
2. Split into 4 pieces
3. Rotate 90° 
4. Apply calibrations
5. **Show GUI and get anchors** ← Main thread
6. Pass anchors to background thread for GIF generation

This adds minimal overhead (~0.1s) and ensures proper threading.

## Compatibility
- **Tested on**: Raspberry Pi OS (Bullseye)
- **Python**: 3.9+
- **Tkinter**: 8.6+
- **Display**: Works with both HDMI displays and touchscreens

## Related Issues
- Manual mode wasn't being enabled due to missing/invisible dialog
- Environment variable option added for headless/automated setups
- Batch processing script already had correct implementation (`-m` flag works)

## Future Improvements
1. Add in-app toggle button (keyboard shortcut) to enable/disable without restart
2. Save preference to config file
3. Add visual indicator in viewfinder when manual mode is active
4. Allow editing anchors after they're selected (before processing)

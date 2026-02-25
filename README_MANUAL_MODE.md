# Manual Alignment Mode - Quick Start

## Problem Fixed
The manual alignment GUI was being called from a background thread, which caused it to fail silently. This has been fixed - the GUI now runs on the main thread before processing begins.

## How to Enable Manual Mode

### Option 1: Dialog at Startup (Default)
When you start the app, you'll see a dialog:
```
Enable manual anchor point selection?

Yes = Manually click alignment points for each photo
No = Automatic template-based alignment
```

**IMPORTANT:** Click **Yes** to enable manual mode. The selection will be logged:
```
============================================================
MANUAL ALIGNMENT MODE ENABLED
============================================================
```

### Option 2: Environment Variable (Recommended for Pi)
Set the environment variable to skip the dialog:

```bash
# Enable manual mode
export WIGGLEGRAM_MANUAL_ALIGN=true
python3 src/app.py

# Or use the helper script
chmod +x start_manual_mode.sh
./start_manual_mode.sh
```

To disable:
```bash
export WIGGLEGRAM_MANUAL_ALIGN=false
python3 src/app.py
```

### Option 3: Add to systemd service (for auto-start)
Edit your systemd service file:
```ini
[Service]
Environment="WIGGLEGRAM_MANUAL_ALIGN=true"
ExecStart=/usr/bin/python3 /home/admin/wigglegram/src/app.py
```

## What Happens When Enabled

1. Take a photo (press shutter button)
2. **Manual alignment GUI appears** showing the first camera view
3. Click the same anchor point on each of the 4 camera views
   - Use a distinctive feature (nose tip, eye center, etc.)
   - Navigate with Next/Previous buttons or arrow keys
   - All 4 points must be selected before you can generate
4. Click **✓ Generate** (or press Enter)
5. Processing continues automatically

## Troubleshooting

### "Manual mode enabled but GUI never appeared"
**Cause:** You were in manual mode but the logs show it used automatic alignment.

**Solution:** This was the bug that's now fixed. Update to the latest code and try again.

### "Dialog doesn't appear at startup"
**Cause:** Display issues on Pi, or the dialog is off-screen.

**Solution:** Use environment variable method instead:
```bash
export WIGGLEGRAM_MANUAL_ALIGN=true
python3 src/app.py
```

### "GUI opens but buttons don't work"
**Cause:** Tkinter compatibility or display driver issues.

**Solution:** Try updating Pillow and tkinter:
```bash
sudo apt-get install python3-tk
pip3 install --upgrade Pillow
```

### "Want to use automatic mode again"
**Solution:** 
- Restart the app and click "No" in the dialog, OR
- `export WIGGLEGRAM_MANUAL_ALIGN=false`, OR
- Remove the environment variable: `unset WIGGLEGRAM_MANUAL_ALIGN`

## Verifying Manual Mode is Active

Check the logs after startup:
```bash
tail -f /home/admin/wigglegram/wigglegram.log
```

You should see:
```
============================================================
MANUAL ALIGNMENT MODE ENABLED
============================================================
Starting app with alignment mode: MANUAL
```

When you take a photo, you should see:
```
Manual alignment enabled - showing anchor selection GUI
```

## Performance Notes

- **Manual mode adds ~10-30 seconds** per photo (time to click 4 points)
- The GUI runs on the main thread, so the app will be unresponsive while you're selecting points
- This is normal and expected behavior
- For batch processing, manual mode is slow but gives better results

## See Also

- [Full Manual Alignment Documentation](MANUAL_ALIGNMENT.md)
- [Batch Processing with Manual Mode](src/scripts/batch_create_gifs.py) - use `-m` flag

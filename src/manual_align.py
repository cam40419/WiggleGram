"""Manual anchor point selection GUI for precise wigglegram alignment."""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


class ManualAlignmentWindow:
    """GUI for manually selecting alignment anchor points on each camera view."""
    
    def __init__(self, images: List[Image.Image]):
        """
        Initialize manual alignment window.
        
        Args:
            images: List of PIL Images (one per camera)
        """
        self.images = images
        self.num_images = len(images)
        self.current_index = 0
        self.points: List[Optional[Tuple[int, int]]] = [None] * self.num_images
        self.result: Optional[List[Optional[Tuple[int, int]]]] = None
        
        # Create window
        self.root = tk.Tk()
        self.root.title("Manual Anchor Point Selection")
        self.root.configure(bg="#1e1e1e")
        
        # State variables
        self.display_scale = 1.0
        self.original_size = images[0].size
        
        # Build UI
        self._build_ui()
        self._load_image(0)
        
    def _build_ui(self):
        """Build the user interface."""
        # Top instruction bar
        self.instruction_frame = tk.Frame(self.root, bg="#2d2d30", height=60)
        self.instruction_frame.pack(side=tk.TOP, fill=tk.X)
        
        instruction_text = (
            "Click to select the same anchor point on each image (e.g., center of subject's nose)\n"
            "Use arrow keys or buttons to navigate. Press Generate when all points are set."
        )
        self.instruction_label = tk.Label(
            self.instruction_frame, text=instruction_text,
            font=("Arial", 11), fg="#ffffff", bg="#2d2d30", pady=10
        )
        self.instruction_label.pack()
        
        # Main canvas for image display
        self.canvas_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(
            self.canvas_frame, bg="#000000", highlightthickness=0,
            cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        
        # Bottom control bar
        self.control_frame = tk.Frame(self.root, bg="#2d2d30", height=80)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Navigation buttons
        button_frame = tk.Frame(self.control_frame, bg="#2d2d30")
        button_frame.pack(pady=10)
        
        self.prev_btn = tk.Button(
            button_frame, text="← Previous", command=self._prev_image,
            font=("Arial", 11), width=12, state=tk.DISABLED
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(
            button_frame, text=f"Image 1/{self.num_images}",
            font=("Arial", 12, "bold"), fg="#ffffff", bg="#2d2d30", width=15
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        self.next_btn = tk.Button(
            button_frame, text="Next →", command=self._next_image,
            font=("Arial", 11), width=12
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.generate_btn = tk.Button(
            button_frame, text="✓ Generate", command=self._confirm_and_close,
            font=("Arial", 11, "bold"), width=12, state=tk.DISABLED,
            bg="#0e639c", fg="#ffffff"
        )
        self.generate_btn.pack(side=tk.LEFT, padx=20)
        
        # Preview thumbnails
        self.preview_frame = tk.Frame(self.control_frame, bg="#2d2d30")
        self.preview_frame.pack(pady=(0, 10))
        
        self.preview_labels = []
        for i in range(self.num_images):
            thumb = self.images[i].copy()
            thumb.thumbnail((80, 80))
            photo = ImageTk.PhotoImage(thumb)
            
            frame = tk.Frame(self.preview_frame, bg="#444444", bd=2, relief=tk.RAISED)
            frame.pack(side=tk.LEFT, padx=3)
            
            label = tk.Label(frame, image=photo, bg="#000000")
            label.image = photo  # Keep reference
            label.pack()
            
            status_label = tk.Label(
                frame, text="✗", font=("Arial", 10, "bold"),
                fg="#ff4444", bg="#444444"
            )
            status_label.pack()
            
            self.preview_labels.append((frame, label, status_label))
        
        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<Return>", lambda e: self._confirm_and_close())
        
    def _load_image(self, index: int):
        """Load and display image at given index."""
        self.current_index = index
        img = self.images[index].copy()
        
        # Scale to fit screen while maintaining aspect ratio
        self.root.update_idletasks()
        canvas_width = max(800, self.canvas.winfo_width())
        canvas_height = max(600, self.canvas.winfo_height())
        
        img_width, img_height = img.size
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        self.display_scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        display_width = int(img_width * self.display_scale)
        display_height = int(img_height * self.display_scale)
        
        img_resized = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_resized)
        
        # Clear and redraw canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo, anchor=tk.CENTER, tags="image"
        )
        
        # Draw existing point if set
        if self.points[index] is not None:
            self._draw_crosshair(self.points[index])
        
        # Update UI
        self._update_status()
        
    def _on_click(self, event):
        """Handle mouse click to select anchor point."""
        # Get click position relative to image
        canvas_coords = self.canvas.coords("image")
        if not canvas_coords:
            return
        
        img_center_x, img_center_y = canvas_coords[0], canvas_coords[1]
        img_width = int(self.images[self.current_index].size[0] * self.display_scale)
        img_height = int(self.images[self.current_index].size[1] * self.display_scale)
        
        # Calculate relative position in display coordinates
        rel_x = event.x - (img_center_x - img_width / 2)
        rel_y = event.y - (img_center_y - img_height / 2)
        
        # Check if click is within image bounds
        if 0 <= rel_x <= img_width and 0 <= rel_y <= img_height:
            # Convert to original image coordinates
            orig_x = int(rel_x / self.display_scale)
            orig_y = int(rel_y / self.display_scale)
            
            self.points[self.current_index] = (orig_x, orig_y)
            self._draw_crosshair((orig_x, orig_y))
            self._update_status()
            
            logger.info("Anchor point set for image %d: (%d, %d)", 
                       self.current_index, orig_x, orig_y)
    
    def _on_motion(self, event):
        """Show magnifier on hover (optional enhancement)."""
        # Could implement magnifier like merge.py if desired
        pass
    
    def _draw_crosshair(self, point: Tuple[int, int]):
        """Draw crosshair at the specified point (in original image coordinates)."""
        orig_x, orig_y = point
        
        # Convert to display coordinates
        canvas_coords = self.canvas.coords("image")
        if not canvas_coords:
            return
        
        img_center_x, img_center_y = canvas_coords[0], canvas_coords[1]
        img_width = int(self.images[self.current_index].size[0] * self.display_scale)
        img_height = int(self.images[self.current_index].size[1] * self.display_scale)
        
        display_x = img_center_x - img_width / 2 + orig_x * self.display_scale
        display_y = img_center_y - img_height / 2 + orig_y * self.display_scale
        
        # Remove old crosshair
        self.canvas.delete("crosshair")
        
        # Draw new crosshair
        size = 20
        width = 3
        
        # White outline
        self.canvas.create_line(
            display_x - size, display_y, display_x + size, display_y,
            fill="white", width=width + 2, tags="crosshair"
        )
        self.canvas.create_line(
            display_x, display_y - size, display_x, display_y + size,
            fill="white", width=width + 2, tags="crosshair"
        )
        
        # Red center
        self.canvas.create_line(
            display_x - size, display_y, display_x + size, display_y,
            fill="red", width=width, tags="crosshair"
        )
        self.canvas.create_line(
            display_x, display_y - size, display_x, display_y + size,
            fill="red", width=width, tags="crosshair"
        )
        
        # Circle at center
        r = 4
        self.canvas.create_oval(
            display_x - r, display_y - r, display_x + r, display_y + r,
            fill="red", outline="white", width=2, tags="crosshair"
        )
    
    def _update_status(self):
        """Update status label and preview thumbnails."""
        # Update status label
        self.status_label.config(text=f"Image {self.current_index + 1}/{self.num_images}")
        
        # Update preview highlights
        for i, (frame, label, status_label) in enumerate(self.preview_labels):
            if i == self.current_index:
                frame.config(bg="#0e639c", relief=tk.RAISED, bd=3)
            else:
                frame.config(bg="#444444", relief=tk.RAISED, bd=2)
            
            # Update checkmark/X status
            if self.points[i] is not None:
                status_label.config(text="✓", fg="#44ff44")
            else:
                status_label.config(text="✗", fg="#ff4444")
        
        # Enable/disable navigation buttons
        self.prev_btn.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_btn.config(
            state=tk.NORMAL if self.current_index < self.num_images - 1 else tk.DISABLED
        )
        
        # Enable generate button if all points are set
        if all(p is not None for p in self.points):
            self.generate_btn.config(state=tk.NORMAL)
        else:
            self.generate_btn.config(state=tk.DISABLED)
    
    def _next_image(self):
        """Navigate to next image."""
        if self.current_index < self.num_images - 1:
            self._load_image(self.current_index + 1)
    
    def _prev_image(self):
        """Navigate to previous image."""
        if self.current_index > 0:
            self._load_image(self.current_index - 1)
    
    def _confirm_and_close(self):
        """Confirm all points are set and close window."""
        if not all(p is not None for p in self.points):
            messagebox.showwarning(
                "Incomplete Selection",
                f"Please set anchor points for all {self.num_images} images."
            )
            return
        
        logger.info("Manual alignment complete: %s", self.points)
        self.result = self.points
        self.root.quit()
        self.root.destroy()
    
    def run(self) -> Optional[List[Optional[Tuple[int, int]]]]:
        """
        Run the GUI and return selected anchor points.
        
        Returns:
            List of (x, y) tuples for each image, or None if cancelled
        """
        try:
            self.root.mainloop()
            return self.result
        except Exception as e:
            logger.error("Manual alignment window error: %s", e)
            return None


def get_manual_anchors(images: List[Image.Image]) -> Optional[List[Optional[Tuple[int, int]]]]:
    """
    Show GUI to manually select anchor points.
    
    Args:
        images: List of PIL Images to align
        
    Returns:
        List of anchor points (x, y) or None if cancelled
    """
    window = ManualAlignmentWindow(images)
    return window.run()

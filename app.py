"""
YOLO Detection GUI (Tkinter)
----------------------------
Place this file next to the `my_model/` folder containing `my_model.pt`
and the `train/` results folder, then run:  python app.py
"""

import os
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# ---------- Config ----------
DEFAULT_MODEL_PATH = Path("my_model/my_model.pt")
TRAIN_DIR = Path("my_model/train")
SUPPORTED_IMAGES = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
    ("All files", "*.*"),
]


class YoloApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Object Detection")
        self.root.geometry("1150x720")
        self.root.minsize(900, 600)

        # State
        self.model: YOLO | None = None
        self.model_path: Path | None = None
        self.current_image_path: str | None = None
        self.original_image: Image.Image | None = None
        self.annotated_image: Image.Image | None = None

        self._build_ui()
        self._try_load_model(DEFAULT_MODEL_PATH)

    # ---------- UI ----------
    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root, padx=10, pady=8)
        toolbar.pack(side="top", fill="x")

        tk.Button(toolbar, text="📂 Open Image", command=self.open_image).pack(side="left", padx=3)
        tk.Button(toolbar, text="🗂 Open Folder (batch)", command=self.process_folder).pack(side="left", padx=3)
        tk.Button(toolbar, text="▶ Run Detection", command=self.run_detection).pack(side="left", padx=3)
        tk.Button(toolbar, text="💾 Save Result", command=self.save_result).pack(side="left", padx=3)
        tk.Button(toolbar, text="📊 Training Metrics", command=self.show_metrics).pack(side="left", padx=3)
        tk.Button(toolbar, text="🔄 Load Different Model", command=self.choose_model).pack(side="left", padx=3)

        # Confidence threshold slider
        tk.Label(toolbar, text="Confidence:").pack(side="left", padx=(20, 4))
        self.conf_var = tk.DoubleVar(value=0.25)
        tk.Scale(
            toolbar, from_=0.05, to=0.95, resolution=0.05,
            orient="horizontal", variable=self.conf_var, length=160, showvalue=True,
        ).pack(side="left")

        # Main area: left = image tabs, right = results
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Left: tabs for Original / Annotated
        left = tk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True)

        self.original_frame = tk.Frame(self.notebook, bg="#1e1e1e")
        self.annotated_frame = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.original_frame, text="Original")
        self.notebook.add(self.annotated_frame, text="Detection")

        self.original_label = tk.Label(self.original_frame, bg="#1e1e1e")
        self.original_label.pack(fill="both", expand=True)
        self.annotated_label = tk.Label(self.annotated_frame, bg="#1e1e1e")
        self.annotated_label.pack(fill="both", expand=True)

        # Re-render on resize so the image fits the window
        self.original_label.bind("<Configure>", lambda e: self._refresh_image(self.original_image, self.original_label))
        self.annotated_label.bind("<Configure>", lambda e: self._refresh_image(self.annotated_image, self.annotated_label))

        # Right: results panel
        right = tk.Frame(main, width=320)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="Detection Results", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.results_text = tk.Text(right, width=38, wrap="word", font=("Consolas", 10))
        self.results_text.pack(fill="both", expand=True, pady=(4, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w").pack(side="bottom", fill="x")

    # ---------- Model ----------
    def _try_load_model(self, path: Path):
        if not path.exists():
            self._set_status(f"Model not found at {path} — use 'Load Different Model'")
            return
        try:
            self.model = YOLO(str(path))
            self.model_path = path
            self._set_status(f"Model loaded: {path}")
        except Exception as e:
            messagebox.showerror("Model load error", str(e))

    def choose_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO model (.pt)",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self._try_load_model(Path(path))

    # ---------- Actions ----------
    def open_image(self):
        path = filedialog.askopenfilename(title="Choose an image", filetypes=SUPPORTED_IMAGES)
        if not path:
            return
        self.current_image_path = path
        self.original_image = Image.open(path).convert("RGB")
        self.annotated_image = None
        self._refresh_image(self.original_image, self.original_label)
        self.annotated_label.configure(image="")
        self.results_text.delete("1.0", "end")
        self.notebook.select(0)
        self._set_status(f"Loaded: {os.path.basename(path)}")

    def run_detection(self):
        if self.model is None:
            messagebox.showerror("No model", "Model is not loaded.")
            return
        if self.original_image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return

        self._set_status("Running detection...")
        self.root.update_idletasks()

        try:
            results = self.model.predict(
                source=self.current_image_path,
                conf=self.conf_var.get(),
                verbose=False,
            )
        except Exception as e:
            messagebox.showerror("Detection error", str(e))
            self._set_status("Error during detection")
            return

        result = results[0]

        # YOLO returns BGR via .plot(); convert to RGB for PIL/Tk
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB) # reads images in BGR (Blue, Green, Red) format by default, this function is essential for converting them to other formats like RGB (for library compatibility like Matplotlib) or Grayscale (for simpler image processing)
        self.annotated_image = Image.fromarray(annotated_rgb)

        self._refresh_image(self.annotated_image, self.annotated_label)
        self.notebook.select(1)
        self._update_results_panel(result)
        self._set_status(f"Done — {len(result.boxes)} object(s) detected")

    def process_folder(self):
        if self.model is None:
            messagebox.showerror("No model", "Model is not loaded.")
            return
        folder = filedialog.askdirectory(title="Pick folder with images")
        if not folder:
            return

        self._set_status("Processing folder... (may take a while)")
        self.root.update_idletasks()
        try:
            results = self.model.predict(
                source=folder,
                conf=self.conf_var.get(),
                save=True,
                verbose=False,
            )
        except Exception as e:
            messagebox.showerror("Detection error", str(e))
            return

        save_dir = results[0].save_dir if results else "?"
        total_objects = sum(len(r.boxes) for r in results)
        messagebox.showinfo(
            "Batch done",
            f"Images processed: {len(results)}\n"
            f"Total detections: {total_objects}\n"
            f"Saved to: {save_dir}",
        )
        self._set_status(f"Batch finished — {len(results)} images")

    def save_result(self):
        if self.annotated_image is None:
            messagebox.showwarning("Nothing to save", "Run detection first.")
            return
        default_name = "detection.jpg"
        if self.current_image_path:
            stem = Path(self.current_image_path).stem
            default_name = f"{stem}_detected.jpg"
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if not path:
            return
        self.annotated_image.save(path)
        self._set_status(f"Saved: {path}")

    def show_metrics(self):
        if not TRAIN_DIR.exists():
            messagebox.showwarning("Not found", f"Folder {TRAIN_DIR} does not exist.")
            return

        # Common YOLO output plots, in a sensible order
        preferred = [
            "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
            "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png",
            "labels.jpg", "labels_correlogram.jpg",
            "val_batch0_pred.jpg", "val_batch0_labels.jpg",
        ]
        found = [f for f in preferred if (TRAIN_DIR / f).exists()]
        # Add any other images we missed
        for p in sorted(TRAIN_DIR.iterdir()):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.name not in found:
                found.append(p.name)

        if not found:
            messagebox.showinfo("No metrics", "No metric images found in train folder.")
            return

        win = tk.Toplevel(self.root)
        win.title("Training Metrics")
        win.geometry("900x650")

        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True)

        for fname in found:
            frame = tk.Frame(notebook, bg="white")
            notebook.add(frame, text=fname)
            img = Image.open(TRAIN_DIR / fname)
            img.thumbnail((860, 580), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(frame, image=photo, bg="white")
            label.image = photo  # prevent garbage collection
            label.pack(expand=True, padx=10, pady=10)

    # ---------- Helpers ----------
    def _update_results_panel(self, result):
        self.results_text.delete("1.0", "end")
        boxes = result.boxes
        names = result.names

        if boxes is None or len(boxes) == 0:
            self.results_text.insert("end", "No objects detected.\n\n")
            self.results_text.insert("end", "Try lowering the confidence threshold.")
            return

        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        confs = boxes.conf.cpu().numpy().tolist()
        counts = Counter(class_ids)

        self.results_text.insert("end", f"Total: {len(boxes)} objects\n")
        self.results_text.insert("end", f"Confidence ≥ {self.conf_var.get():.2f}\n")
        self.results_text.insert("end", "─" * 30 + "\n")

        self.results_text.insert("end", "By class:\n")
        for cid, n in counts.most_common():
            self.results_text.insert("end", f"  • {names[cid]:<15} × {n}\n")

        self.results_text.insert("end", "\nDetections:\n")
        for i, (cid, conf) in enumerate(zip(class_ids, confs), 1):
            self.results_text.insert("end", f"{i:>3}. {names[cid]:<15} {conf:>6.1%}\n")

    def _refresh_image(self, pil_img: Image.Image | None, label: tk.Label):
        if pil_img is None:
            return
        w = max(label.winfo_width(), 50)
        h = max(label.winfo_height(), 50)
        if w < 50 or h < 50:
            w, h = 700, 500
        img = pil_img.copy()
        img.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo  # keep reference

    def _set_status(self, text: str):
        self.status_var.set(text)


if __name__ == "__main__":
    root = tk.Tk()
    YoloApp(root)
    root.mainloop()

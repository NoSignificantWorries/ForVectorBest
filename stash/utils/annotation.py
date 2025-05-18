import os
import glob
import json
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np


def resize_image_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    orig_width, orig_height = image.size
    ratio = min(max_width / orig_width, max_height / orig_height)
    new_size = (int(orig_width * ratio), int(orig_height * ratio))
    return image.resize(new_size, Image.LANCZOS)


class ImageObj:
    def __init__(self, image_path: str, ksize: int):
        self.image_path = image_path
        self.image = None
        self.origin = None
        self.origin_size = None
        self.pil_image = None
        self.photo_image = None
        self.ksize = ksize
        
        self.annotation = {}
    
    def pad_image(self) -> None:
        if self.origin is None:
            return

        H, W, _ = self.origin.shape
        self.origin_size = (H, W)
        
        new_H, new_W = (H // self.ksize + 1) * self.ksize, (W // self.ksize + 1) * self.ksize
        h_pad, w_pad = (new_H - H) // 2, (new_W - W) // 2

        padded_img = np.zeros((new_H, new_W, 3), dtype=np.uint8)
        padded_img[h_pad:h_pad + H, w_pad:w_pad + W, :] = self.origin
        
        self.image = padded_img

    def load(self) -> None:
        self.origin = cv2.imread(self.image_path)
        self.origin = cv2.cvtColor(self.origin, cv2.COLOR_BGR2RGB)
        
        self.pad_image()

        self.pil_image = Image.fromarray(self.image)
        self.photo_image = ImageTk.PhotoImage(image=self.pil_image)
    
    def set_ksize(self, ksize: int):
        self.ksize = ksize
        self.pad_image()
    
    def set_annatation(self, annotation: dict):
        self.annotation = annotation
    
    def set_class(self, cls: int, ksize: int, rect_id: int):
        if str(ksize) not in self.annotation.keys():
            self.annotation[str(ksize)] = {}
        self.annotation[str(ksize)][str(rect_id)] = cls

    def get_annotation(self) -> dict:
        return self.annotation
    
    def get(self) -> np.ndarray | None:
        return self.image
    
    def get_pil(self):
        return self.pil_image

    def get_photo(self):
        return self.photo_image

    def get_origin_size(self) -> tuple[int, int] | None:
        return self.origin_size

    def get_size(self) -> tuple[int, int] | None:
        if self.image is None:
            return None
        return self.image.shape[0], self.image.shape[1]
    
    def get_name(self):
        return self.image_path


annotation = {}
images = []
idx = 0
cnt = 0
current_image = None

options = ["512", "256", "128", "64", "32", "16", "8", "4"]
base_ksize = 512
current_ksize = base_ksize
lines = []
canvas = None
cell_rects = {}
current_rect_id = -1
canvas_rect_id = -1
classes = [1, 2, 3, 4]
# 0 - base
# 1 - dirt
# 2 - didn't print
# 3 - blurred
class_colors = {
    1: (255, 0, 0, 120),    # Red, semi-transparent
    2: (0, 255, 0, 120),    # Green
    3: (0, 0, 255, 120),    # Blue
    4: (255, 255, 0, 120),  # Yellow
    5: (255, 0, 255, 120),  # Purple
    6: (0, 255, 255, 120),  # Cyan
    7: (255, 128, 0, 120),  # Orange
    8: (128, 0, 255, 120),  # Lavender
    9: (0, 0, 0, 120),      # Black
}

root = tk.Tk()

selected_option = tk.StringVar()
selected_option.set(options[0])

root.title("Image Annotator 3000 PRO MEGA ULTRA")
root.geometry("1920x1080")
root.configure(bg="#1e1e2e")

style = ttk.Style()
style.theme_use('clam')

style.configure("RoundedButton.TButton",
                background="#1e1e2e",
                foreground="#cdd6f4",
                font=("Arial", 14, "bold"),
                padding=10,
                relief="ridge")

style.map("RoundedButton.TButton",
          background=[("active", "!disabled", "#11111b")],
          foreground=[("active", "#a6adc8")],
          relief=[("pressed", "sunken")])

def on_canvas_click(event):
    global canvas, current_image, base_ksize, cell_rects, current_rect_id, canvas_rect_id, current_ksize

    if not current_image or not canvas:
        return

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    pil_img = current_image.get_pil()
    if pil_img is None:
        return

    orig_width, orig_height = pil_img.size
    scale_ratio = min(canvas_width / orig_width, canvas_height / orig_height)
    resized_width = int(orig_width * scale_ratio)
    resized_height = int(orig_height * scale_ratio)
    x_offset = (canvas_width - resized_width) // 2
    y_offset = (canvas_height - resized_height) // 2

    x_img = event.x - x_offset
    y_img = event.y - y_offset

    if not (0 <= x_img < resized_width and 0 <= y_img < resized_height):
        return

    scaled_ksize = int(current_ksize * scale_ratio)
    if scaled_ksize <= 0:
        return

    col = x_img // scaled_ksize
    row = y_img // scaled_ksize

    x0 = x_offset + col * scaled_ksize
    y0 = y_offset + row * scaled_ksize
    x1 = x0 + scaled_ksize
    y1 = y0 + scaled_ksize

    cell_id = row * (current_image.get_size()[1] // current_ksize) + col
    if current_rect_id != cell_id:
        canvas.delete(canvas_rect_id)
        current_rect_id = cell_id
        canvas_rect_id = canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2, outline="red", width=3)
    else:
        canvas.delete(canvas_rect_id)
        current_rect_id = -1
        canvas_rect_id = -1


def on_key_press(event):
    global current_rect_id, classes, current_image, current_ksize
    
    if current_image is None:
        return
    if current_rect_id < 0:
        return
    if event.char not in "123456789":
        return
    
    cls = int(event.char)
    if cls in classes:
        current_image.set_class(cls, current_ksize, current_rect_id)
    
    print(current_image.get_annotation())
    draw_image_on_canvas()


def create_canvas(shape: tuple[int, int]):
    global canvas
    if canvas is not None:
        canvas.destroy()
    canvas = tk.Canvas(root, bg="#1e1e2e")
    canvas.grid(row=1, column=0, sticky="nsew")
    canvas.bind("<Configure>", draw_image_on_canvas)
    canvas.bind("<Button-1>", on_canvas_click)
    canvas.bind("<Key>", on_key_press)
    canvas.focus_set()


def open_dir():
    global images, idx, cnt, current_image, base_ksize
    annotation = {}
    
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        if os.path.exists(os.path.join(folder_selected, "annotation.json")):
            with open(os.path.join(folder_selected, "annotation.json"), "r", encoding="utf-8") as file:
                annotation = json.load(file)

        images = []
        idx = 0
        image_pathes = glob.glob(os.path.join(folder_selected, "*.jpg")) + glob.glob(os.path.join(folder_selected, "*.png"))
        for file in image_pathes:
            img = ImageObj(file, base_ksize)
            if os.path.basename(file) in annotation.keys():
                img.set_annatation(annotation[os.path.basename(file)])
            images.append(img)
        cnt = len(images)
        print(f"{cnt} files loaded")
        view_img()


def draw_image_on_canvas(event=None):
    global canvas, current_image, base_ksize, cell_rects, current_ksize, class_colors
    if not current_image or not canvas:
        return
    
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    
    if canvas_width <= 1 or canvas_height <= 1:
        return
        
    pil_img = current_image.get_pil()
    if pil_img is None:
        return
        
    resized_img = resize_image_to_fit(pil_img, canvas_width, canvas_height)
    photo_img = ImageTk.PhotoImage(resized_img)
    
    canvas.photo_img = photo_img
    
    orig_width, orig_height = pil_img.size
    scale_ratio = min(canvas_width / orig_width, canvas_height / orig_height)
    
    x_offset = (canvas_width - resized_img.width) // 2
    y_offset = (canvas_height - resized_img.height) // 2
    
    canvas.delete("all")
    cell_rects.clear()
    canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo_img)
    
    scaled_ksize = int(current_ksize * scale_ratio)
    if scaled_ksize > 0:
        for i in range(x_offset, x_offset + resized_img.width, scaled_ksize):
            canvas.create_line(i, y_offset, i, y_offset + resized_img.height, fill="gray")
        for j in range(y_offset, y_offset + resized_img.height, scaled_ksize):
            canvas.create_line(x_offset, j, x_offset + resized_img.width, j, fill="gray")
    
    annotation = current_image.get_annotation()

    overlay = Image.new("RGBA", (resized_img.width, resized_img.height), (0, 0, 0, 0))
    for ksize_str, rects in annotation.items():
        if ksize_str != str(current_ksize):
            continue
        for rect_id_str, cls in rects.items():
            try:
                rect_id = int(rect_id_str)
                cls = int(cls)
            except Exception:
                continue
            num_cols = current_image.get_size()[1] // current_ksize
            row = rect_id // num_cols
            col = rect_id % num_cols
            
            x0 = x_offset + col * scaled_ksize
            y0 = y_offset + row * scaled_ksize
            x1 = x0 + scaled_ksize
            y1 = y0 + scaled_ksize
            
            draw = ImageDraw.Draw(overlay, "RGBA")
            draw.rectangle([x0-x_offset, y0-y_offset, x1-x_offset, y1-y_offset], fill=class_colors[cls], outline=None)

    if overlay.getbbox():
        overlay_tk = ImageTk.PhotoImage(overlay)
        canvas.overlay_img = overlay_tk
        canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=overlay_tk)


def view_img():
    global images, idx, cnt, current_image, base_ksize, label, controls, current_rect_id, canvas_rect_id

    current_rect_id = -1
    canvas_rect_id = -1

    current_image = images[idx]
    current_image.load()
    create_canvas(current_image.get_size())
    draw_image_on_canvas()
    label.configure(text=f"Image {idx + 1} of {cnt}")


def prev_img():
    global images, idx, cnt, current_image

    if not bool(images):
        return
    idx = ((idx - 1) + cnt) % cnt
    view_img()


def next_img():
    global images, idx, cnt, current_image

    if not bool(images):
        return
    idx = (idx + 1) % cnt
    view_img()


def get_selected_value():
    global selected_option, current_ksize
    new_ksize = int(selected_option.get())
    
    if new_ksize != current_ksize:
        current_ksize = new_ksize
        draw_image_on_canvas()
    if canvas is not None:
        canvas.focus_set()


def save_annotation():
    global images
    
    if not bool(images):
        return
    
    path = images[0].get_name()
    path = os.path.split(path)[0]
    
    res = {}
    for image in images:
        if bool(image.get_annotation()):
            res[os.path.basename(image.get_name())] = image.get_annotation()
    with open(os.path.join(path, "annotation.json"), "w", encoding="utf-8") as file:
        json.dump(res, file, indent=4)


controls = tk.Frame(root, background="#1e1e2e")
controls.grid(row=0, column=0, pady=30, sticky="ew")

button = ttk.Button(controls, text="Open dir", style="RoundedButton.TButton", command=open_dir)
button.grid(row=0, column=0, padx=5, pady=5)
prev_btn = ttk.Button(controls, text="< prev", style="RoundedButton.TButton", command=prev_img)
prev_btn.grid(row=0, column=4, padx=5, pady=5, sticky=tk.E)
next_btn = ttk.Button(controls, text="next >", style="RoundedButton.TButton", command=next_img)
next_btn.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
get_value = ttk.Button(controls, text="Save", style="RoundedButton.TButton", command=save_annotation)
get_value.grid(row=0, column=6, padx=5, pady=5)

label = ttk.Label(controls, text="Select any floder", background="#1e1e2e", foreground="#cdd6f4", font=("Arial", 18, "bold"))
label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

get_value = ttk.Button(controls, text="Set", style="RoundedButton.TButton", command=get_selected_value)
get_value.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
combobox = ttk.Combobox(controls, textvariable=selected_option, values=options, state="readonly")
combobox.configure(font=("Arial", 20, "bold"))
combobox.grid(row=0, column=2, padx=5, pady=5, sticky=tk.E)

controls.columnconfigure(0, weight=1)
controls.columnconfigure(1, weight=1)
controls.columnconfigure(2, weight=1)
controls.columnconfigure(3, weight=1)
controls.columnconfigure(4, weight=1)
controls.columnconfigure(5, weight=1)
controls.columnconfigure(6, weight=1)

root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.mainloop()

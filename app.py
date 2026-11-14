"""OCT Lens — просмотр ОКТ-снимков и классификация с MobileNetV2."""

import queue
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from torchvision import models, transforms


BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
NUM_CLASSES = 4
IMG_SIZE = 224
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_DESCRIPTIONS = {
    "CNV": "Хориоидальная неоваскуляризация",
    "DME": "Диабетический макулярный отёк",
    "DRUSEN": "Друзы сетчатки",
    "NORMAL": "Класс нормы",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
inference_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

BG = "#171C1E"
PANEL = "#202729"
SURFACE = "#2A3436"
CANVAS = "#101516"
TEXT = "#EEF2EE"
MUTED = "#A9B8B7"
ACCENT = "#A3C9BE"
HOVER = "#BEDBCF"
ERROR = "#E3A597"


def load_model(model_path=MODEL_PATH) -> nn.Module:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError("Рядом с программой отсутствует файл best_model.pth.")
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return model.eval()


def predict(model: nn.Module, image_path):
    device = next(model.parameters()).device
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    input_tensor = inference_transforms(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor), dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    return CLASS_NAMES[predicted_idx.item()], confidence.item() * 100


class OCTClassifierApp(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        super().__init__()
        self.title("OCT Lens")
        width = min(1120, self.winfo_screenwidth() - 40)
        height = min(740, self.winfo_screenheight() - 90)
        self.geometry(f"{width}x{height}")
        self.minsize(860, 560)
        self.configure(fg_color=BG)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_image_path = None
        self.source_image = None
        self.photo_image = None
        self._busy = True
        self._events = queue.Queue()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._poll_id = self.after(40, self._poll)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(22, 18))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header, text="OCT Lens", text_color=TEXT,
            font=ctk.CTkFont(size=27, weight="bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text="Классификация ОКТ-снимков сетчатки",
            text_color=MUTED, font=ctk.CTkFont(size=13), anchor="w",
        ).grid(row=1, column=0, sticky="w")
        ctk.CTkLabel(
            header, text=f"MobileNetV2   ·   {self.device.type.upper()}",
            text_color=MUTED, font=ctk.CTkFont(size=12),
        ).grid(row=0, column=1, rowspan=2, sticky="e")
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=28)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, minsize=280)
        main.grid_rowconfigure(0, weight=1)
        image_panel = ctk.CTkFrame(main, fg_color=PANEL, corner_radius=8)
        image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        image_panel.grid_columnconfigure(0, weight=1)
        image_panel.grid_rowconfigure(1, weight=1)
        self.file_label = ctk.CTkLabel(
            image_panel, text="Снимок", anchor="w", width=1,
            text_color=TEXT, font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.file_label.grid(row=0, column=0, sticky="ew", padx=16, pady=(10, 6))
        self.canvas = ctk.CTkCanvas(image_panel, bg=CANVAS, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.canvas.bind("<Configure>", self._redraw_preview)
        result = ctk.CTkFrame(main, width=280, fg_color=PANEL, corner_radius=8)
        result.grid(row=0, column=1, sticky="nsew")
        result.grid_propagate(False)
        result.grid_columnconfigure(0, weight=1)
        result.grid_rowconfigure(7, weight=1)
        ctk.CTkLabel(
            result, text="Результат модели", anchor="w", text_color=TEXT,
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 22))
        ctk.CTkLabel(
            result, text="Предсказанный класс", anchor="w", text_color=MUTED,
            font=ctk.CTkFont(size=12),
        ).grid(row=1, column=0, sticky="ew", padx=22)
        self.class_label = ctk.CTkLabel(
            result, text="—", anchor="w", text_color=TEXT,
            font=ctk.CTkFont(size=32, weight="bold"),
        )
        self.class_label.grid(row=2, column=0, sticky="ew", padx=22, pady=(2, 2))
        self.desc_label = ctk.CTkLabel(
            result, text="Загрузите снимок и запустите анализ.",
            anchor="nw", justify="left", wraplength=230, text_color=MUTED,
            font=ctk.CTkFont(size=13),
        )
        self.desc_label.grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 30))
        ctk.CTkLabel(
            result, text="Уверенность модели", anchor="w", text_color=MUTED,
            font=ctk.CTkFont(size=12),
        ).grid(row=4, column=0, sticky="ew", padx=22)
        self.conf_label = ctk.CTkLabel(
            result, text="—", anchor="w", text_color=ACCENT,
            font=ctk.CTkFont(size=25, weight="bold"),
        )
        self.conf_label.grid(row=5, column=0, sticky="ew", padx=22, pady=(2, 8))
        self.conf_bar = ctk.CTkProgressBar(
            result, height=5, corner_radius=2, fg_color=SURFACE, progress_color=ACCENT,
        )
        self.conf_bar.grid(row=6, column=0, sticky="ew", padx=22)
        self.conf_bar.set(0)
        ctk.CTkLabel(
            result, text="Учебная модель. Результат требует оценки специалиста.",
            anchor="w", justify="left", wraplength=230, text_color=MUTED,
            font=ctk.CTkFont(size=12),
        ).grid(row=8, column=0, sticky="ew", padx=22, pady=20)
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=28, pady=(18, 22))
        footer.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(
            footer, text="Загрузка модели…", anchor="w", text_color=MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.load_btn = ctk.CTkButton(
            footer, text="Открыть снимок", command=self._load_image,
            fg_color=SURFACE, hover_color="#3C494B", text_color=TEXT,
            height=40, width=160, corner_radius=6, state="disabled",
        )
        self.load_btn.grid(row=0, column=1, padx=(12, 10))
        self.predict_btn = ctk.CTkButton(
            footer, text="Анализировать", command=self._predict,
            fg_color=ACCENT, hover_color=HOVER, text_color=CANVAS,
            height=40, width=160, corner_radius=6, state="disabled",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.predict_btn.grid(row=0, column=2)

    def _load_model(self):
        try:
            model = load_model().to(self.device)
            self._events.put(("model", model))
        except Exception as exc:
            self._events.put(("model_error", str(exc)))

    def _load_image(self):
        if self._busy or self.model is None:
            return
        path = filedialog.askopenfilename(
            title="Открыть ОКТ-снимок",
            filetypes=[("Изображения", "*.jpeg *.jpg *.png *.bmp *.tif *.tiff")],
        )
        if not path:
            return
        try:
            with Image.open(path) as image:
                loaded = image.convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            messagebox.showerror("Не удалось открыть снимок", str(exc), parent=self)
            return
        self.source_image = loaded
        self.current_image_path = path
        self.file_label.configure(text=Path(path).name)
        self._redraw_preview()
        self.class_label.configure(text="—")
        self.desc_label.configure(text="Снимок готов к анализу.")
        self.conf_label.configure(text="—")
        self.conf_bar.set(0)
        self.predict_btn.configure(state="normal")
        self.status_label.configure(text=f"{loaded.width} × {loaded.height} пикс.", text_color=MUTED)

    def _redraw_preview(self, _event=None):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if min(width, height) < 4:
            return
        self.canvas.delete("all")
        if self.source_image is None:
            self.canvas.create_text(
                width // 2, height // 2, text="Откройте ОКТ-снимок",
                fill=MUTED, font=("Segoe UI", 15),
            )
            return
        image = self.source_image.copy()
        image.thumbnail((width - 2, height - 2), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(width // 2, height // 2, image=self.photo_image)

    def _predict(self):
        if self._busy or self.model is None or self.source_image is None:
            return
        self._busy = True
        self.load_btn.configure(state="disabled")
        self.predict_btn.configure(state="disabled", text="Анализ…")
        self.class_label.configure(text="—")
        self.desc_label.configure(text="Обработка снимка…")
        self.conf_label.configure(text="—")
        self.conf_bar.set(0)
        self.status_label.configure(text="Анализ снимка…", text_color=MUTED)
        image = self.source_image.copy()

        def worker():
            try:
                self._events.put(("prediction", predict(self.model, image)))
            except Exception as exc:
                self._events.put(("prediction_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            kind, payload = self._events.get_nowait()
        except queue.Empty:
            self._poll_id = self.after(40, self._poll)
            return
        self._busy = False
        if kind == "model":
            self.model = payload
            self.load_btn.configure(state="normal")
            self.status_label.configure(text="Модель готова")
        elif kind == "model_error":
            self.status_label.configure(text="Модель недоступна", text_color=ERROR)
            messagebox.showerror("Не удалось загрузить модель", payload, parent=self)
        else:
            self.load_btn.configure(state="normal")
            self.predict_btn.configure(state="normal", text="Анализировать")
            if kind == "prediction":
                class_name, confidence = payload
                self.class_label.configure(text=class_name)
                self.desc_label.configure(text=CLASS_DESCRIPTIONS[class_name])
                self.conf_label.configure(text=f"{confidence:.2f}%")
                self.conf_bar.set(confidence / 100)
                self.status_label.configure(text="Анализ завершён")
            else:
                self.desc_label.configure(text="Попробуйте выполнить анализ ещё раз.")
                self.status_label.configure(text="Ошибка анализа", text_color=ERROR)
                messagebox.showerror("Не удалось обработать снимок", payload, parent=self)
        self._poll_id = self.after(40, self._poll)

    def _close(self):
        self.after_cancel(self._poll_id)
        self.destroy()


if __name__ == "__main__":
    OCTClassifierApp().mainloop()

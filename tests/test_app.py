import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import app


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = app.load_model()

    def test_missing_weights_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                app.load_model(Path(folder) / "missing.pth")

    def test_invalid_weights_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.pth"
            torch.save({"invalid": torch.ones(1)}, path)
            with self.assertRaises(RuntimeError):
                app.load_model(path)

    def test_path_and_loaded_image_match(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "снимок.png"
            image = Image.new("L", (420, 220), 100)
            image.save(path)
            expected = app.predict(self.model, image)
            actual = app.predict(self.model, path)
            self.assertEqual(expected, actual)
            self.assertIn(actual[0], app.CLASS_NAMES)
            self.assertTrue(0 <= actual[1] <= 100)


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.errors = []
        cls.error_patch = patch.object(app.messagebox, "showerror", side_effect=lambda *a, **k: cls.errors.append(a))
        cls.error_patch.start()
        cls.window = app.OCTClassifierApp()
        cls.addClassCleanup(cls.error_patch.stop)
        cls.addClassCleanup(cls.window._close)

    def setUp(self):
        self.wait(lambda: not self.window._busy)
        self.assertIsNotNone(self.window.model)
        self.errors.clear()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "снимок.png"
        Image.new("RGB", (320, 180), "#808080").save(self.path)

    def wait(self, condition):
        deadline = time.monotonic() + 20
        while not condition() and time.monotonic() < deadline:
            self.window.update()
            time.sleep(.01)
        self.assertTrue(condition())
        self.window.update()

    def open_image(self, path):
        with patch.object(app.filedialog, "askopenfilename", return_value=str(path)):
            self.window._load_image()

    def test_analyse_and_reset_on_new_image(self):
        self.open_image(self.path)
        self.window._predict()
        self.assertEqual(self.window.load_btn.cget("state"), "disabled")
        self.wait(lambda: not self.window._busy)
        self.assertIn(self.window.class_label.cget("text"), app.CLASS_NAMES)
        self.assertEqual(self.window.load_btn.cget("state"), "normal")
        self.open_image(self.path)
        self.assertEqual(self.window.class_label.cget("text"), "—")
        self.assertEqual(self.window.conf_bar.get(), 0)
        self.assertFalse(self.errors)

    def test_invalid_image_preserves_current_image(self):
        self.open_image(self.path)
        broken = self.path.with_name("broken.png")
        broken.write_bytes(b"invalid image")
        self.open_image(broken)
        self.assertEqual(self.window.current_image_path, str(self.path))
        self.assertEqual(len(self.errors), 1)
        self.assertEqual(self.window.predict_btn.cget("state"), "normal")

    def test_prediction_error_allows_retry(self):
        self.open_image(self.path)
        with patch.object(app, "predict", side_effect=RuntimeError("Test error")):
            self.window._predict()
            self.wait(lambda: not self.window._busy)
        self.assertEqual(self.window.class_label.cget("text"), "—")
        self.assertEqual(self.window.predict_btn.cget("state"), "normal")
        self.assertEqual(len(self.errors), 1)
        self.window._predict()
        self.wait(lambda: not self.window._busy)
        self.assertIn(self.window.class_label.cget("text"), app.CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()

import contextlib
import copy
import glob
import os
import re
import tkinter as tk
import traceback

from modules.modelSampler.BaseModelSampler import ModelSamplerOutput
from modules.ui.SampleFrame import SampleFrame
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.FileType import FileType
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
from PIL import Image


class PairBuilderWindow(ctk.CTkToplevel):
    """Interactive Pair Builder: generates two images with the same prompt and different seeds, then dumps chosen/rejected via Pick A/B."""

    def __init__(
            self,
            parent,
            train_config: TrainConfig,
            callbacks: TrainCallbacks | None = None,
            commands: TrainCommands | None = None,
            train_ui=None,
            *args, **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.title("Pair Builder — Interactive DPO")
        self.geometry("1300x820")
        self.resizable(True, True)

        self.train_config = train_config
        self.callbacks = callbacks
        self.commands = commands
        self.train_ui = train_ui

        model_type = train_config.model_type
        self.sample = SampleConfig.default_values(model_type)
        self.ui_state = UIState(self, self.sample)
        self._train_config_ui_state = UIState(self, self.train_config)

        # External model (training active) mode: register callback
        if callbacks is not None:
            self.callbacks.set_on_sample_custom(self.__on_image_received)
            self.callbacks.set_on_update_sample_custom_progress(self.__on_sample_progress)

        # Image slot state
        self._original_image_a: Image.Image | None = None
        self._original_image_b: Image.Image | None = None
        self._pending_slot: str | None = None   # "A" or "B"

        # Session counter
        self._session_pair_count: int = 0

        # Resize debounce after id
        self._resize_after_id: str | None = None

        self.__build_ui(model_type)

        if self.train_ui is not None:
            self.train_ui.add_training_state_listener(self.__on_training_state_changed)
            cb, cmd = self.train_ui.get_current_runtime()
            if cmd is not None:
                self.__on_training_state_changed("running")
            else:
                self.__on_training_state_changed("idle")

        self.__update_pair_count_label()

        self.wait_visibility()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))
        self.bind("<Configure>", self.__on_resize)

        if self.train_ui is not None:
            self.train_ui.lock_main_start_button()

    # ──────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────

    def __build_ui(self, model_type):
        self.grid_rowconfigure(0, weight=0)   # prompt
        self.grid_rowconfigure(1, weight=0)   # settings
        self.grid_rowconfigure(2, weight=0)   # action buttons
        self.grid_rowconfigure(3, weight=1, minsize=400)   # image area
        self.grid_rowconfigure(4, weight=0)   # pick buttons
        self.grid_rowconfigure(5, weight=0)   # status bar (pair counter)
        self.grid_rowconfigure(6, weight=0)   # interactive settings
        self.grid_columnconfigure(0, weight=1, minsize=500)
        self.grid_columnconfigure(1, weight=1, minsize=500)

        # Prompt + settings (two SampleFrame instances)
        prompt_frame = SampleFrame(self, self.sample, self.ui_state, include_settings=False, model_type=model_type)
        prompt_frame.grid(row=0, column=0, columnspan=2, padx=0, pady=0, sticky="nsew")

        settings_frame = SampleFrame(self, self.sample, self.ui_state, include_prompt=False, model_type=model_type)
        settings_frame.grid(row=1, column=0, columnspan=2, padx=0, pady=0, sticky="nsew")

        # Action buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, columnspan=2, padx=0, pady=4, sticky="nsew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        self._generate_btn = ctk.CTkButton(btn_frame, text="Generate Pair", command=self.__generate_pair)
        self._generate_btn.grid(row=0, column=0, padx=10, pady=4, sticky="ew")

        self._train_button = ctk.CTkButton(
            btn_frame, text="Start Training",
            command=self.__on_train_button_clicked,
        )
        self._train_button.grid(row=0, column=1, padx=10, pady=4, sticky="ew")

        # Image display labels (left/right)
        dummy = self.__dummy_image()

        self.ctk_image_a = ctk.CTkImage(light_image=dummy, size=(200, 200))
        self.ctk_image_b = ctk.CTkImage(light_image=dummy, size=(200, 200))

        # Container A — fixed-size cell; text label on top, image label centered below
        self._image_container_a = ctk.CTkFrame(self, fg_color="transparent")
        self._image_container_a.grid(row=3, column=0, padx=8, pady=4, sticky="nsew")
        self._image_container_a.grid_propagate(False)
        self._image_container_a.grid_rowconfigure(0, weight=0)
        self._image_container_a.grid_rowconfigure(1, weight=1)
        self._image_container_a.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self._image_container_a, text="A: Current LoRA", text_color="gray").grid(row=0, column=0, sticky="ew", pady=(2, 0))
        self.image_label_a = ctk.CTkLabel(self._image_container_a, text="", image=self.ctk_image_a)
        self.image_label_a.grid(row=1, column=0)

        # Container B
        self._image_container_b = ctk.CTkFrame(self, fg_color="transparent")
        self._image_container_b.grid(row=3, column=1, padx=8, pady=4, sticky="nsew")
        self._image_container_b.grid_propagate(False)
        self._image_container_b.grid_rowconfigure(0, weight=0)
        self._image_container_b.grid_rowconfigure(1, weight=1)
        self._image_container_b.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self._image_container_b, text="B: Reference (base/frozen)", text_color="gray").grid(row=0, column=0, sticky="ew", pady=(2, 0))
        self.image_label_b = ctk.CTkLabel(self._image_container_b, text="", image=self.ctk_image_b)
        self.image_label_b.grid(row=1, column=0)

        # Pick buttons
        self._pick_a_btn = ctk.CTkButton(self, text="Pick A", command=lambda: self.__pick("A"))
        self._pick_a_btn.grid(row=4, column=0, padx=10, pady=4, sticky="ew")

        self._pick_b_btn = ctk.CTkButton(self, text="Pick B", command=lambda: self.__pick("B"))
        self._pick_b_btn.grid(row=4, column=1, padx=10, pady=4, sticky="ew")

        # Row 5: progress bar + status label + pair counter in one frame
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=6, sticky="ew")
        status_frame.grid_columnconfigure(2, weight=1)

        self._progress_bar = ctk.CTkProgressBar(status_frame, width=200)
        self._progress_bar.set(0)
        self._progress_bar.grid(row=0, column=0, padx=(0, 8), pady=2, sticky="w")

        self._progress_status_label = ctk.CTkLabel(status_frame, text="", text_color="gray", anchor="w")
        self._progress_status_label.grid(row=0, column=1, padx=(0, 16), pady=2, sticky="w")

        self._pair_counter_label = ctk.CTkLabel(status_frame, text="Session: 0   Total: 0", anchor="e")
        self._pair_counter_label.grid(row=0, column=2, padx=(0, 0), pady=2, sticky="e")

        # Interactive DPO settings row
        settings_row_frame = ctk.CTkFrame(self, fg_color="transparent")
        settings_row_frame.grid(row=6, column=0, columnspan=2, padx=6, pady=(0, 6), sticky="ew")
        settings_row_frame.grid_columnconfigure(1, weight=1)

        cleanup_label = ctk.CTkLabel(settings_row_frame, text="Cleanup on Stop:")
        cleanup_label.grid(row=0, column=0, padx=(4, 2), pady=2, sticky="w")
        self._cleanup_switch = ctk.CTkSwitch(
            settings_row_frame, text="",
            command=self.__on_cleanup_toggled,
        )
        self._cleanup_switch.grid(row=0, column=1, padx=(0, 4), pady=2, sticky="w")
        if self.train_config.rlhf_interactive_cleanup_on_stop:
            self._cleanup_switch.select()
        else:
            self._cleanup_switch.deselect()

    # ──────────────────────────────────────────────
    # Pair generation
    # ──────────────────────────────────────────────

    def __generate_pair(self):
        """Enqueue two images with the same prompt but different seeds and cfg_scale offsets to increase distinguishability."""
        import random
        self._original_image_a = None
        self._original_image_b = None
        self._pending_slot = "A"

        if self.commands is None:
            from tkinter import messagebox
            messagebox.showinfo(
                "Pair Builder",
                "Start Training first — pairs can only be generated while training is running or in Ready state.",
                parent=self,
            )
            return

        self._progress_bar.set(0)
        self._progress_status_label.configure(text="Generating A...")

        sample_a = copy.copy(self.sample)
        sample_b = copy.copy(self.sample)

        # If the user fixed a seed (random_seed=False), force B to a different one so A and B aren't identical.
        # When random_seed=True (default), the trainer picks fresh random seeds for each call — leave them alone.
        if not sample_a.random_seed:
            sample_b.seed = (sample_a.seed + random.randint(1, 10000)) % (2**31 - 1)

        # Spread cfg_scale ±delta around base to make A/B visually distinct
        base_cfg = self.sample.cfg_scale
        delta = random.uniform(0.3, 1.0)
        sample_a.cfg_scale = max(base_cfg + delta, 1.0)
        sample_b.cfg_scale = max(base_cfg - delta, 1.0)

        # Fixed mapping: A = current LoRA (trained side), B = reference (base or frozen snapshot).
        # Trainer-oriented view — the user wants to track training direction, not blind-label.
        sample_b.use_reference_model = True

        self.commands.sample_custom(sample_a)
        self.commands.sample_custom(sample_b)

    def __on_image_received(self, sampler_output: ModelSamplerOutput):
        """Trainer callback — fills A/B slots in order."""
        if sampler_output.file_type != FileType.IMAGE:
            return
        image: Image.Image = sampler_output.data

        if self._pending_slot == "A":
            self._original_image_a = image
            self.__display_image("A", image)
            self._pending_slot = "B"
        elif self._pending_slot == "B":
            self._original_image_b = image
            self.__display_image("B", image)
            self._pending_slot = None
            # Both slots complete — reset progress
            self._progress_bar.set(0)
            self._progress_status_label.configure(text="")

    def __on_sample_progress(self, progress: int, max_progress: int):
        """Called by trainer on each sampling step. Thread-safe: dispatch to main thread."""
        self.after(0, lambda: self.__apply_sample_progress(progress, max_progress))

    def __apply_sample_progress(self, progress: int, max_progress: int):
        if max_progress <= 0:
            return
        ratio = progress / max_progress
        self._progress_bar.set(ratio)
        slot_label = self._pending_slot or ""
        if slot_label:
            self._progress_status_label.configure(text=f"Generating {slot_label}... ({progress}/{max_progress})")
        else:
            self._progress_status_label.configure(text=f"Generating... ({progress}/{max_progress})")

    # ──────────────────────────────────────────────
    # Image display (aspect-ratio-preserving fit)
    # ──────────────────────────────────────────────

    def __display_image(self, slot: str, pil_image: Image.Image):
        """Resize the image to fit the container area while preserving aspect ratio."""
        container = self._image_container_a if slot == "A" else self._image_container_b
        label = self.image_label_a if slot == "A" else self.image_label_b

        self.update()
        # CTkImage applies HiDPI widget scaling on top of the size we pass, so we must
        # divide by the current widget scaling to compute a fitted pixel size that
        # actually fits within container.winfo_width() (which is in logical pixels).
        try:
            scaling = ctk.ScalingTracker.get_widget_scaling(self)
        except (AttributeError, Exception):
            scaling = 1.0
        max_w = max(int(container.winfo_width() * 0.92 / scaling), 200)
        # Subtract ~30px for the "A:/B:" text label row at the top of the container.
        max_h = max(int((container.winfo_height() - 30) * 0.92 / scaling), 200)

        ratio = min(max_w / pil_image.width, max_h / pil_image.height)
        new_w = max(int(pil_image.width * ratio), 1)
        new_h = max(int(pil_image.height * ratio), 1)

        fitted = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        new_ctk_img = ctk.CTkImage(light_image=fitted, size=(new_w, new_h))
        if slot == "A":
            self.ctk_image_a = new_ctk_img
        else:
            self.ctk_image_b = new_ctk_img
        label.configure(image=new_ctk_img)

    def __on_resize(self, event):
        """Debounce window resize events and re-render images."""
        if self._resize_after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(120, self.__redraw_images)

    def __redraw_images(self):
        if self._original_image_a:
            self.__display_image("A", self._original_image_a)
        if self._original_image_b:
            self.__display_image("B", self._original_image_b)

    def __clear_image_slots(self):
        dummy = self.__dummy_image()
        self.ctk_image_a.configure(light_image=dummy, size=(200, 200))
        self.ctk_image_b.configure(light_image=dummy, size=(200, 200))

    def __dummy_image(self) -> Image.Image:
        return Image.new("RGB", (512, 512), color=(30, 30, 30))

    # ──────────────────────────────────────────────
    # Pick → folder dump
    # ──────────────────────────────────────────────

    def __pick(self, chosen_slot: str):
        """Save the selected slot as chosen and the other as rejected into pair_NNNNN files."""
        if self._original_image_a is None or self._original_image_b is None:
            from tkinter import messagebox
            messagebox.showwarning("Pair Builder", "Generate two images with Generate Pair first.", parent=self)
            return

        chosen = self._original_image_a if chosen_slot == "A" else self._original_image_b
        rejected = self._original_image_b if chosen_slot == "A" else self._original_image_a

        pairs_dir = self.train_config.rlhf_interactive_pairs_dir
        if not pairs_dir:
            from tkinter import messagebox
            messagebox.showerror("Pair Builder", "Interactive Pairs Folder must be set in the RLHF tab first.", parent=self)
            return

        chosen_dir = os.path.join(pairs_dir, "chosen")
        rejected_dir = os.path.join(pairs_dir, "rejected")
        os.makedirs(chosen_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

        next_id = self.__next_pair_id(chosen_dir, rejected_dir)
        stem = f"pair_{next_id:05d}"

        chosen.save(os.path.join(chosen_dir, stem + ".png"))
        rejected.save(os.path.join(rejected_dir, stem + ".png"))

        prompt_text = self.sample.prompt or ""
        for d in (chosen_dir, rejected_dir):
            with open(os.path.join(d, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(prompt_text)

        self._session_pair_count += 1
        self.__update_pair_count_label()

        # Reset slots
        self._original_image_a = None
        self._original_image_b = None
        self.__clear_image_slots()

    def __next_pair_id(self, chosen_dir: str, rejected_dir: str) -> int:
        """Return max pair_NNNNN.png number across both folders plus 1. Returns 1 if none exist."""
        pattern = re.compile(r"pair_(\d+)\.png$", re.IGNORECASE)
        max_id = 0
        for directory in (chosen_dir, rejected_dir):
            for path in glob.glob(os.path.join(directory, "pair_*.png")):
                m = pattern.search(os.path.basename(path))
                if m:
                    max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    # ──────────────────────────────────────────────
    # Training state sync
    # ──────────────────────────────────────────────

    def __update_pair_count_label(self):
        """Refresh the status bar label with session count and folder total."""
        pairs_dir = self.train_config.rlhf_interactive_pairs_dir or ""
        chosen_dir = os.path.join(pairs_dir, "chosen") if pairs_dir else ""
        total = len(glob.glob(os.path.join(chosen_dir, "pair_*.png"))) if chosen_dir else 0
        self._pair_counter_label.configure(
            text=f"Session: {self._session_pair_count}   Total: {total}"
        )

    def __on_cleanup_toggled(self):
        """Sync the switch state back to train_config."""
        self.train_config.rlhf_interactive_cleanup_on_stop = bool(self._cleanup_switch.get())

    def __on_train_button_clicked(self):
        if self.train_ui is None:
            from tkinter import messagebox
            messagebox.showerror("Training", "TrainUI reference not available", parent=self)
            return
        self.train_ui.start_training()

    def __on_training_state_changed(self, state: str):
        if state == "running":
            cb, cmd = self.train_ui.get_current_runtime() if self.train_ui else (None, None)
            self.callbacks = cb
            self.commands = cmd
            if self.callbacks is not None:
                self.callbacks.set_on_sample_custom(self.__on_image_received)
                self.callbacks.set_on_update_sample_custom_progress(self.__on_sample_progress)
            self._train_button.configure(
                text="Stop",
                state="normal",
                fg_color="#dc3545",
                hover_color="#bb2d3b",
            )
            self._generate_btn.configure(state="normal")
        elif state == "waiting":
            # Training thread is alive but paused — refresh callbacks/commands so Generate Pair can talk to it
            cb, cmd = self.train_ui.get_current_runtime() if self.train_ui else (None, None)
            self.callbacks = cb
            self.commands = cmd
            if self.callbacks is not None:
                self.callbacks.set_on_sample_custom(self.__on_image_received)
                self.callbacks.set_on_update_sample_custom_progress(self.__on_sample_progress)
            self._train_button.configure(
                text="Ready",
                state="normal",
                fg_color="#0d6efd",
                hover_color="#0b5ed7",
            )
            self._generate_btn.configure(state="normal")
        elif state == "stopping":
            self._train_button.configure(text="Stopping...", state="disabled")
            self._generate_btn.configure(state="disabled")
        elif state == "idle":
            if self.callbacks is not None:
                self.callbacks.set_on_sample_custom()
                self.callbacks.set_on_update_sample_custom_progress()
            self.callbacks = None
            self.commands = None
            self._train_button.configure(
                text="Start Training",
                state="normal",
                fg_color="#198754",
                hover_color="#146c43",
            )
            self._generate_btn.configure(state="disabled")

    # ──────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────

    def destroy(self):
        try:
            if self.train_ui is not None:
                try:
                    self.train_ui.unlock_main_start_button()
                except Exception:
                    pass
                self.train_ui.remove_training_state_listener(self.__on_training_state_changed)

            if hasattr(self, "_icon_image_ref"):
                del self._icon_image_ref

            if self._resize_after_id is not None:
                with contextlib.suppress(tk.TclError):
                    self.after_cancel(self._resize_after_id)

            for after_id in self.tk.call("after", "info"):
                with contextlib.suppress(tk.TclError, RuntimeError):
                    self.after_cancel(after_id)

            super().destroy()
        except (tk.TclError, RuntimeError) as e:
            print(f"Error destroying PairBuilderWindow: {e}")
        except Exception as e:
            print(f"Unexpected error destroying PairBuilderWindow: {e}")
            traceback.print_exc()

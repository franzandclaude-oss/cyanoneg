"""cyanoneg GUI: Process and Profiles tabs (Calibrate arrives in Phase 2).

Deliberately plain tkinter — the value of this tool is in the pipeline, not the chrome.
Processing runs on a worker thread so the window stays responsive; results and errors come
back to the UI via ``after`` polling of a queue.
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from tkinter import (
    BOTH,
    DISABLED,
    END,
    LEFT,
    NORMAL,
    RIGHT,
    E,
    StringVar,
    Tk,
    W,
    X,
    filedialog,
    messagebox,
    ttk,
)
from tkinter.scrolledtext import ScrolledText

import numpy as np
from PIL import Image as PILImage
from PIL import ImageTk

from .. import imageio as cio
from ..mono import DEFAULT_WEIGHTS, channel_noise, suggest_weights
from ..pipeline import DEFAULT_OUTPUT_PPI, PrintSize, make_negative
from ..profiles import PROFILE_DIR, Profile, list_profiles
from ..targets import blocker_grid, exposure_strip, step_wedge

PREVIEW_MAX = 420


class App:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("cyanoneg — cyanotype digital negatives")
        root.geometry("980x640")

        self.queue: queue.Queue = queue.Queue()
        self.profiles: dict[str, Profile] = {}
        self.preview_photo: ImageTk.PhotoImage | None = None

        notebook = ttk.Notebook(root)
        notebook.pack(fill=BOTH, expand=True)
        self.process_tab = ttk.Frame(notebook, padding=10)
        self.profiles_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.process_tab, text="Process")
        notebook.add(self.profiles_tab, text="Profiles")

        self._build_process_tab()
        self._build_profiles_tab()
        self._reload_profiles()
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------ profiles state

    def _reload_profiles(self) -> None:
        self.profiles.clear()
        errors = []
        for path in list_profiles():
            try:
                self.profiles[path.stem] = Profile.load(path)
            except Exception as e:  # noqa: BLE001 - surface every unloadable profile
                errors.append(f"{path.name}: {e}")
        names = sorted(self.profiles)
        self.profile_box["values"] = names
        if names and self.profile_var.get() not in names:
            self.profile_var.set(names[0])
        self._refresh_profile_list()
        self._on_profile_selected()
        if errors:
            messagebox.showwarning("Profiles", "Some profiles failed to load:\n" + "\n".join(errors))

    def _current_profile(self) -> Profile | None:
        return self.profiles.get(self.profile_var.get())

    # ------------------------------------------------------------------ Process tab

    def _build_process_tab(self) -> None:
        tab = self.process_tab
        left = ttk.Frame(tab)
        left.pack(side=LEFT, fill=BOTH, expand=False, padx=(0, 10))
        right = ttk.LabelFrame(tab, text="Preview (print orientation is mirrored — this is the film)")
        right.pack(side=RIGHT, fill=BOTH, expand=True)

        self.preview_label = ttk.Label(right, anchor="center")
        self.preview_label.pack(fill=BOTH, expand=True, padx=6, pady=6)

        row = 0
        grid = ttk.Frame(left)
        grid.pack(fill=X)

        def add_row(label: str) -> int:
            nonlocal row
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky=W, pady=3)
            row += 1
            return row - 1

        # Source
        r = add_row("Source positive")
        self.source_var = StringVar()
        ttk.Entry(grid, textvariable=self.source_var, width=42).grid(row=r, column=1, sticky=W + E)
        ttk.Button(grid, text="…", width=3, command=self._pick_source).grid(row=r, column=2, padx=3)

        r = add_row("Colour space")
        self.space_var = StringVar(value="auto")
        ttk.Combobox(
            grid, textvariable=self.space_var, values=("auto", *cio.SPACES), state="readonly", width=12
        ).grid(row=r, column=1, sticky=W)

        self.raw_var = StringVar(value="0")
        ttk.Checkbutton(
            grid, text="Raw scan (un-inverted negative)", variable=self.raw_var, onvalue="1", offvalue="0"
        ).grid(row=row, column=1, sticky=W, pady=3)
        row += 1

        # Profile
        r = add_row("Profile")
        self.profile_var = StringVar()
        self.profile_box = ttk.Combobox(grid, textvariable=self.profile_var, state="readonly", width=32)
        self.profile_box.grid(row=r, column=1, sticky=W)
        self.profile_box.bind("<<ComboboxSelected>>", lambda _e: self._on_profile_selected())

        self.profile_note = ttk.Label(grid, text="", foreground="#a06000", wraplength=340)
        self.profile_note.grid(row=row, column=1, sticky=W)
        row += 1

        # Size / ppi
        r = add_row("Print size (mm)")
        size = ttk.Frame(grid)
        size.grid(row=r, column=1, sticky=W)
        self.width_var, self.height_var = StringVar(value="240"), StringVar(value="180")
        ttk.Entry(size, textvariable=self.width_var, width=7).pack(side=LEFT)
        ttk.Label(size, text=" × ").pack(side=LEFT)
        ttk.Entry(size, textvariable=self.height_var, width=7).pack(side=LEFT)
        ttk.Label(size, text="  ppi ").pack(side=LEFT)
        self.ppi_var = StringVar(value=str(DEFAULT_OUTPUT_PPI))
        ttk.Entry(size, textvariable=self.ppi_var, width=6).pack(side=LEFT)

        # Weights
        r = add_row("Channel weights R,G,B")
        weights = ttk.Frame(grid)
        weights.grid(row=r, column=1, sticky=W)
        self.weights_var = StringVar(value=",".join(str(w) for w in DEFAULT_WEIGHTS))
        ttk.Entry(weights, textvariable=self.weights_var, width=16).pack(side=LEFT)
        ttk.Button(weights, text="Analyse noise", command=self._analyse_noise).pack(side=LEFT, padx=4)

        # Output
        r = add_row("Output TIFF")
        self.output_var = StringVar()
        ttk.Entry(grid, textvariable=self.output_var, width=42).grid(row=r, column=1, sticky=W + E)
        ttk.Button(grid, text="…", width=3, command=self._pick_output).grid(row=r, column=2, padx=3)

        self.process_button = ttk.Button(left, text="Make negative", command=self._process)
        self.process_button.pack(fill=X, pady=(10, 4))

        self.status = ttk.Label(left, text="", wraplength=380, foreground="#205020")
        self.status.pack(fill=X)

    def _pick_source(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose positive scan",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if path:
            self.source_var.set(path)
            if not self.output_var.get():
                p = Path(path)
                self.output_var.set(str(p.with_name(p.stem + "_negative.tif")))

    def _pick_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save negative as", defaultextension=".tif", filetypes=[("TIFF", "*.tif")]
        )
        if path:
            self.output_var.set(path)

    def _on_profile_selected(self) -> None:
        profile = self._current_profile()
        if profile is None:
            self.profile_note.config(text="")
            return
        notes = []
        if profile.provisional:
            notes.append("PROVISIONAL — not measured; tones are a starting point, not a calibration.")
        if not profile.is_ready_to_print:
            notes.append("No blocker colour yet: print and read the HSB grid before using this profile.")
        self.profile_note.config(text=" ".join(notes))

    def _parse_weights(self) -> tuple[float, float, float]:
        parts = [float(v) for v in self.weights_var.get().split(",")]
        if len(parts) != 3:
            raise ValueError("weights must be three comma-separated numbers")
        return (parts[0], parts[1], parts[2])

    def _analyse_noise(self) -> None:
        source = self.source_var.get().strip()
        if not source:
            messagebox.showinfo("Analyse noise", "Choose a source image first.")
            return

        def work() -> None:
            try:
                space = None if self.space_var.get() == "auto" else self.space_var.get()
                image = cio.load_image(source, space=space)  # type: ignore[arg-type]
                if image.is_mono:
                    self.queue.put(("info", "Source is already monochrome — nothing to analyse."))
                    return
                noise = channel_noise(image)
                suggested = suggest_weights(noise)
                report = "   ".join(f"{ch}: {v:.4f}" for ch, v in noise.items())
                self.queue.put(("noise", (report, suggested)))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("error", f"Noise analysis failed: {e}"))

        threading.Thread(target=work, daemon=True).start()
        self.status.config(text="Analysing channel noise…")

    def _process(self) -> None:
        source = self.source_var.get().strip()
        output = self.output_var.get().strip()
        profile = self._current_profile()
        if not source or profile is None or not output:
            messagebox.showinfo("Process", "Choose a source image, a profile, and an output path.")
            return
        try:
            width, height = float(self.width_var.get()), float(self.height_var.get())
            ppi = float(self.ppi_var.get())
            weights = self._parse_weights()
        except ValueError as e:
            messagebox.showerror("Process", f"Check the numeric fields: {e}")
            return
        if not profile.is_ready_to_print:
            messagebox.showerror(
                "Process",
                f"Profile {profile.name!r} has no blocker colour yet.\n"
                "Print and measure the HSB blocker grid first, or use another profile.",
            )
            return

        space = None if self.space_var.get() == "auto" else self.space_var.get()
        raw = self.raw_var.get() == "1"
        self.process_button.config(state=DISABLED)
        self.status.config(text="Processing…")

        def work() -> None:
            try:
                negative = make_negative(
                    source,
                    profile,
                    PrintSize(width, height),
                    output_path=output,
                    output_ppi=ppi,
                    weights=weights,
                    space=space,  # type: ignore[arg-type]
                    raw_scan=raw,
                )
                self.queue.put(("done", (output, negative.data)))
            except cio.ColourSpaceError as e:
                self.queue.put(("error", f"{e}\n\nSet 'Colour space' explicitly and retry."))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("error", f"{e}\n\n{traceback.format_exc(limit=3)}"))

        threading.Thread(target=work, daemon=True).start()

    def _show_preview(self, data: np.ndarray) -> None:
        arr = (np.clip(data, 0.0, 1.0) * 255).astype(np.uint8)
        pil = PILImage.fromarray(arr)
        pil.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
        self.preview_photo = ImageTk.PhotoImage(pil)
        self.preview_label.config(image=self.preview_photo)

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "done":
                    output, data = payload
                    self.process_button.config(state=NORMAL)
                    self.status.config(text=f"Saved {output}")
                    self._show_preview(data)
                elif kind == "noise":
                    report, suggested = payload
                    self.status.config(text=f"Noise σ  {report}")
                    if messagebox.askyesno(
                        "Channel noise",
                        f"Noise σ per channel:\n{report}\n\n"
                        f"Suggested weights: {suggested}\nApply them?",
                    ):
                        self.weights_var.set(",".join(str(w) for w in suggested))
                elif kind == "info":
                    self.process_button.config(state=NORMAL)
                    self.status.config(text=str(payload))
                elif kind == "error":
                    self.process_button.config(state=NORMAL)
                    self.status.config(text="")
                    messagebox.showerror("cyanoneg", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------ Profiles tab

    def _build_profiles_tab(self) -> None:
        tab = self.profiles_tab
        left = ttk.Frame(tab)
        left.pack(side=LEFT, fill=BOTH, padx=(0, 10))

        ttk.Label(left, text=f"Profiles in {PROFILE_DIR}").pack(anchor=W)
        self.profile_list = ttk.Treeview(left, columns=("state",), show="tree headings", height=16)
        self.profile_list.heading("#0", text="Profile")
        self.profile_list.heading("state", text="State")
        self.profile_list.column("#0", width=240)
        self.profile_list.column("state", width=110)
        self.profile_list.pack(fill=BOTH, expand=True, pady=4)
        self.profile_list.bind("<<TreeviewSelect>>", lambda _e: self._show_profile_details())

        buttons = ttk.Frame(left)
        buttons.pack(fill=X, pady=4)
        ttk.Button(buttons, text="Reload", command=self._reload_profiles).pack(side=LEFT)
        ttk.Button(buttons, text="Validate", command=self._validate_selected).pack(side=LEFT, padx=4)

        targets = ttk.LabelFrame(left, text="Generate calibration targets (print via linear profile)")
        targets.pack(fill=X, pady=6)
        ttk.Button(targets, text="Exposure strip", command=lambda: self._generate_target("exposure")).pack(
            side=LEFT, padx=4, pady=4
        )
        ttk.Button(targets, text="Blocker grid", command=lambda: self._generate_target("grid")).pack(
            side=LEFT, padx=4
        )
        ttk.Button(targets, text="256-step wedge", command=lambda: self._generate_target("wedge")).pack(
            side=LEFT, padx=4
        )

        self.detail = ScrolledText(tab, width=58, state=DISABLED, font=("Consolas", 9))
        self.detail.pack(side=RIGHT, fill=BOTH, expand=True)

    def _refresh_profile_list(self) -> None:
        self.profile_list.delete(*self.profile_list.get_children())
        for name, profile in sorted(self.profiles.items()):
            if profile.provisional:
                state = "provisional"
            elif profile.lut.is_identity():
                state = "linear baseline"
            else:
                state = "measured"
            if not profile.is_ready_to_print:
                state += ", no blocker"
            self.profile_list.insert("", END, iid=name, text=name, values=(state,))

    def _selected_profile(self) -> Profile | None:
        selection = self.profile_list.selection()
        return self.profiles.get(selection[0]) if selection else None

    def _show_profile_details(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            return
        import json

        d = profile.to_dict()
        d["lut"] = {
            "size": profile.lut.size,
            "identity": profile.lut.is_identity(),
            "values": "…omitted…" if not profile.lut.is_identity() else "identity",
        }
        text = json.dumps(d, indent=2)
        self.detail.config(state=NORMAL)
        self.detail.delete("1.0", END)
        self.detail.insert("1.0", text)
        self.detail.config(state=DISABLED)

    def _validate_selected(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            messagebox.showinfo("Validate", "Select a profile first.")
            return
        problems = profile.validate()
        if problems:
            messagebox.showwarning("Validate", f"{profile.name}:\n- " + "\n- ".join(problems))
        else:
            messagebox.showinfo("Validate", f"{profile.name}: valid.")

    def _generate_target(self, kind: str) -> None:
        profile = self._selected_profile() or self._current_profile()
        rgb = (255, 0, 0)
        sat = 1.0
        used_placeholder = True
        if profile is not None and profile.is_ready_to_print:
            rgb = tuple(profile.blocker["rgb"])  # type: ignore[assignment]
            sat = float(profile.blocker["saturation"])
            used_placeholder = False

        out = Path("targets")
        try:
            if kind == "exposure":
                target = exposure_strip(blocker_rgb=rgb)  # type: ignore[arg-type]
            elif kind == "grid":
                target = blocker_grid()
                used_placeholder = False  # the grid sweeps its own colours
            else:
                target = step_wedge(rgb, saturation=sat)  # type: ignore[arg-type]
            tif, side = target.save(out)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Targets", str(e))
            return
        note = (
            "\n\nNote: used placeholder red — measure the blocker grid first for real calibration."
            if used_placeholder and kind != "grid"
            else ""
        )
        messagebox.showinfo("Targets", f"Wrote {tif}\nand {side}.{note}")


def main() -> int:
    root = Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:  # noqa: BLE001 - theme is cosmetic
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

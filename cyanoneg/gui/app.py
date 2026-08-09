"""cyanoneg GUI: Process, Profiles, Calibrate and Batch tabs.

Deliberately plain tkinter — the value of this tool is in the pipeline, not the chrome.
Processing runs on a worker thread so the window stays responsive; results and errors come
back to the UI via ``after`` polling of a queue.

Every colour, font and metric comes from :mod:`cyanoneg.gui.theme`; none are written
inline here, so restyling the interface is a change to that one module.

The Calibrate tab walks the three-step wizard from PLAN.md: exposure (record SPE) →
blocker (read the HSB grid scan) → linearisation (read the wedge scan, save the measured
profile and export .cube/.acv). The Process tab can preview either the film or a soft
proof of the predicted print, the latter only where the profile carries measurements.
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
from ..analyze import GridAnalysis, WedgeAnalysis, analyze_grid, analyze_wedge
from ..blocker import hue_to_rgb
from ..mono import DEFAULT_WEIGHTS, channel_noise, suggest_weights
from ..imageio import Image
from ..pipeline import DEFAULT_OUTPUT_PPI, PrintSize, batch_negatives, make_negative
from ..proof import can_proof, soft_proof
from ..profiles import PROFILE_DIR, Profile, list_profiles
from ..targets import blocker_grid, exposure_strip, step_wedge
from . import theme


def output_for_source(source: str | Path, chosen: str = "") -> str:
    """Where the negative for `source` should go.

    The filename always comes from the source image, so processing a second picture
    cannot silently overwrite the first one's negative.

    ``chosen`` is a destination Steven picked or typed himself, and *only* then is its
    folder kept — a name meant for one image should not carry over to another, but a
    folder he deliberately chose should. Without one the negative lands beside its own
    source, so images from different folders do not all pile into the first one's.
    """
    source = Path(source)
    name = source.stem + "_negative.tif"
    return str(Path(chosen).with_name(name) if chosen else source.with_name(name))


class App:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("cyanoneg — cyanotype digital negatives")
        theme.apply(root)

        self.queue: queue.Queue = queue.Queue()
        self.profiles: dict[str, Profile] = {}
        self.preview_photo: ImageTk.PhotoImage | None = None

        notebook = ttk.Notebook(root)
        notebook.pack(fill=BOTH, expand=True)
        self.process_tab = ttk.Frame(notebook, padding=theme.PAD)
        self.profiles_tab = ttk.Frame(notebook, padding=theme.PAD)
        self.calibrate_tab = ttk.Frame(notebook, padding=theme.PAD)
        self.batch_tab = ttk.Frame(notebook, padding=theme.PAD)
        notebook.add(self.process_tab, text="Process")
        notebook.add(self.profiles_tab, text="Profiles")
        notebook.add(self.calibrate_tab, text="Calibrate")
        notebook.add(self.batch_tab, text="Batch")

        self._build_process_tab()
        self._build_profiles_tab()
        self._build_calibrate_tab()
        self._build_batch_tab()
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
        self.batch_profile_box["values"] = names
        if names and self.batch_profile_var.get() not in names:
            self.batch_profile_var.set(self.profile_var.get() or names[0])
        self.cal_profile_box["values"] = names
        if names and self.cal_profile_var.get() not in names:
            # Calibration is for the paper profile being measured, not the linear baseline.
            non_linear = [n for n in names if not self.profiles[n].lut.is_identity() or self.profiles[n].provisional]
            self.cal_profile_var.set(non_linear[0] if non_linear else names[0])
        self._refresh_profile_list()
        self._on_profile_selected()
        self._refresh_wizard_state()
        if errors:
            messagebox.showwarning("Profiles", "Some profiles failed to load:\n" + "\n".join(errors))

    def _current_profile(self) -> Profile | None:
        return self.profiles.get(self.profile_var.get())

    # ------------------------------------------------------------------ Process tab

    @staticmethod
    def _form_row(parent, label: str, row: int):
        """Label in a fixed-width column so every form lines up down the app."""
        ttk.Label(parent, text=label, width=theme.LABEL_WIDTH, anchor=W).grid(
            row=row, column=0, sticky=W, pady=theme.ROW_GAP
        )
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=1, sticky=W + E, pady=theme.ROW_GAP)
        return holder

    def _build_process_tab(self) -> None:
        tab = self.process_tab
        left = ttk.Frame(tab)
        left.pack(side=LEFT, fill=BOTH, expand=False, padx=(0, theme.WIDE_GAP))
        right = ttk.Frame(tab)
        right.pack(side=RIGHT, fill=BOTH, expand=True)

        # ---- preview, given the room it needs ----
        header = ttk.Frame(right)
        header.pack(fill=X)
        ttk.Label(header, text="Preview", style="Heading.TLabel").pack(side=LEFT)
        self.preview_mode = StringVar(value="film")
        for value, label in (("proof", "Soft proof"), ("film", "Film (mirrored)")):
            ttk.Radiobutton(
                header, text=label, value=value, variable=self.preview_mode, command=self._redraw_preview
            ).pack(side=RIGHT, padx=(theme.WIDE_GAP, 0))

        self.proof_note = ttk.Label(right, text="", style="Warn.TLabel", wraplength=520)

        stage = ttk.Frame(right, style="Chrome.TFrame")
        stage.pack(fill=BOTH, expand=True, pady=theme.GAP)
        self.preview_label = ttk.Label(
            stage, anchor="center", background=theme.CANVAS_BG, text="No negative yet", foreground=theme.TEXT_MUTED
        )
        self.preview_label.pack(fill=BOTH, expand=True)

        self.preview_caption = ttk.Label(right, text="", style="Mono.TLabel")
        self.preview_caption.pack(fill=X)

        # ---- source ----
        source_box = ttk.LabelFrame(left, text="Source", padding=theme.GROUP_PAD)
        source_box.pack(fill=X)
        source_box.columnconfigure(1, weight=1)

        holder = self._form_row(source_box, "Positive scan", 0)
        self.source_var = StringVar()
        ttk.Entry(holder, textvariable=self.source_var, width=theme.FIELD_WIDTH).pack(side=LEFT)
        ttk.Button(holder, text="…", width=3, style="Small.TButton", command=self._pick_source).pack(
            side=LEFT, padx=(theme.GAP, 0)
        )

        holder = self._form_row(source_box, "Colour space", 1)
        self.space_var = StringVar(value="auto")
        ttk.Combobox(
            holder, textvariable=self.space_var, values=("auto", *cio.SPACES), state="readonly", width=12
        ).pack(side=LEFT)

        self.raw_var = StringVar(value="0")
        ttk.Checkbutton(
            source_box,
            text="Raw scan (un-inverted negative)",
            variable=self.raw_var,
            onvalue="1",
            offvalue="0",
        ).grid(row=2, column=1, sticky=W)

        holder = self._form_row(source_box, "Channel weights", 3)
        self.weights_var = StringVar(value=",".join(str(w) for w in DEFAULT_WEIGHTS))
        ttk.Entry(holder, textvariable=self.weights_var, width=14).pack(side=LEFT)
        ttk.Button(holder, text="Analyse noise", style="Small.TButton", command=self._analyse_noise).pack(
            side=LEFT, padx=(theme.GAP, 0)
        )

        # ---- profile ----
        profile_box = ttk.LabelFrame(left, text="Profile", padding=theme.GROUP_PAD)
        profile_box.pack(fill=X, pady=(theme.GAP, 0))
        profile_box.columnconfigure(1, weight=1)

        holder = self._form_row(profile_box, "Paper profile", 0)
        self.profile_var = StringVar()
        self.profile_box = ttk.Combobox(holder, textvariable=self.profile_var, state="readonly", width=30)
        self.profile_box.pack(side=LEFT)
        self.profile_box.bind("<<ComboboxSelected>>", lambda _e: self._on_profile_selected())

        self.profile_note = ttk.Label(profile_box, text="", style="Warn.TLabel", wraplength=380)
        self.profile_note.grid(row=1, column=0, columnspan=2, sticky=W)
        self.profile_note.grid_remove()  # takes no space until it has something to say

        # ---- output ----
        output_box = ttk.LabelFrame(left, text="Output", padding=theme.GROUP_PAD)
        output_box.pack(fill=X, pady=(theme.GAP, 0))
        output_box.columnconfigure(1, weight=1)

        holder = self._form_row(output_box, "Print size (mm)", 0)
        self.width_var, self.height_var = StringVar(value="240"), StringVar(value="180")
        ttk.Entry(holder, textvariable=self.width_var, width=theme.NARROW_WIDTH).pack(side=LEFT)
        ttk.Label(holder, text="×").pack(side=LEFT, padx=theme.GAP)
        ttk.Entry(holder, textvariable=self.height_var, width=theme.NARROW_WIDTH).pack(side=LEFT)
        ttk.Label(holder, text="at").pack(side=LEFT, padx=theme.GAP)
        self.ppi_var = StringVar(value=str(DEFAULT_OUTPUT_PPI))
        ttk.Entry(holder, textvariable=self.ppi_var, width=6).pack(side=LEFT)
        ttk.Label(holder, text="ppi", style="Muted.TLabel").pack(side=LEFT, padx=(theme.GAP, 0))

        self.auto_orient_var = StringVar(value="1")
        ttk.Checkbutton(
            output_box,
            text="Match orientation to the image",
            variable=self.auto_orient_var,
            onvalue="1",
            offvalue="0",
        ).grid(row=1, column=1, sticky=W)

        holder = self._form_row(output_box, "Save to", 2)
        self.output_var = StringVar()
        # Anything in the box that this app did not put there was chosen by Steven, and
        # his folder is then kept as the source changes. Without this distinction an
        # auto-derived folder looks identical to a chosen one, and every negative ends up
        # in whichever folder the first image happened to live in.
        self._auto_output = ""  # the last value written here by the app itself
        self._chosen_output = ""  # a destination picked with … or typed by hand
        self.output_var.trace_add("write", self._note_output_edited)
        ttk.Entry(holder, textvariable=self.output_var, width=theme.FIELD_WIDTH).pack(side=LEFT)
        ttk.Button(holder, text="…", width=3, style="Small.TButton", command=self._pick_output).pack(
            side=LEFT, padx=(theme.GAP, 0)
        )

        self.size_note = ttk.Label(output_box, text="", style="Mono.TLabel")
        self.size_note.grid(row=3, column=0, columnspan=2, sticky=W, pady=(theme.ROW_GAP, 0))

        for var in (self.source_var, self.width_var, self.height_var, self.ppi_var, self.auto_orient_var):
            var.trace_add("write", lambda *_: self._update_size_note())

        self.process_button = ttk.Button(
            left, text="Make negative", style="Accent.TButton", command=self._process
        )
        self.process_button.pack(fill=X, pady=(theme.WIDE_GAP, theme.GAP))

        self.status = ttk.Label(left, text="", wraplength=420, style="OK.TLabel")
        self.status.pack(fill=X)

    def _update_size_note(self) -> None:
        """Show the print size this image will actually come out at, before committing."""
        source = self.source_var.get().strip()
        if not source or not Path(source).is_file():
            self.size_note.config(text="")
            return
        try:
            box = PrintSize(float(self.width_var.get()), float(self.height_var.get()))
            ppi = float(self.ppi_var.get())
            with PILImage.open(source) as im:
                src_w, src_h = im.size
        except Exception:  # noqa: BLE001 - a half-typed field is not an error worth showing
            self.size_note.config(text="")
            return
        if self.auto_orient_var.get() == "1":
            box = box.oriented_for(src_w, src_h)
        scale = box.fit_scale(src_w, src_h, ppi)
        self.size_note.config(
            text=f"→ prints {src_w * scale / ppi * 25.4:.0f} × {src_h * scale / ppi * 25.4:.0f} mm"
        )

    def _pick_source(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose positive scan",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if path:
            self.source_var.set(path)
            self._set_output(output_for_source(path, self._chosen_output))

    def _set_output(self, value: str) -> None:
        """Write the box ourselves, without it counting as Steven's own choice."""
        self._auto_output = value
        self.output_var.set(value)

    def _note_output_edited(self, *_) -> None:
        value = self.output_var.get()
        if value and value != self._auto_output:
            self._chosen_output = value

    def _pick_output(self) -> None:
        current = self.output_var.get().strip()
        where = {}
        if current:  # open the dialog where the box already points, not at the last cwd
            where = {"initialdir": str(Path(current).parent), "initialfile": Path(current).name}
        path = filedialog.asksaveasfilename(
            title="Save negative as",
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif")],
            **where,
        )
        if path:
            self.output_var.set(path)  # traced, so this counts as Steven's own choice

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
        self.profile_note.grid() if notes else self.profile_note.grid_remove()

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
                    auto_orient=self.auto_orient_var.get() == "1",
                )
                self.queue.put(("done", (output, negative)))
            except cio.ColourSpaceError as e:
                self.queue.put(("error", f"{e}\n\nSet 'Colour space' explicitly and retry."))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("error", f"{e}\n\n{traceback.format_exc(limit=3)}"))

        threading.Thread(target=work, daemon=True).start()

    def _show_preview(self, negative: Image) -> None:
        self._last_negative = negative
        self._redraw_preview()

    def _redraw_preview(self) -> None:
        negative = getattr(self, "_last_negative", None)
        if negative is None:
            return
        profile = self._current_profile()
        data = negative.data
        self.proof_note.config(text="")
        self.proof_note.pack_forget()

        if self.preview_mode.get() == "proof":
            if profile is None or not can_proof(profile):
                self.proof_note.config(
                    text="No measured response in this profile — calibrate first; "
                    "a proof without measurements would be invented."
                )
                self.proof_note.pack(fill=X, pady=(theme.GAP, 0), before=self.preview_label.master)
                self.preview_mode.set("film")
            else:
                data = soft_proof(negative, profile).data

        arr = (np.clip(data, 0.0, 1.0) * 255).astype(np.uint8)
        pil = PILImage.fromarray(arr)
        full_h, full_w = data.shape[:2]
        pil.thumbnail((theme.PREVIEW_MAX, theme.PREVIEW_MAX))
        self.preview_photo = ImageTk.PhotoImage(pil)
        self.preview_label.config(image=self.preview_photo, text="")
        mm = ""
        if negative.ppi:
            mm = f"   {full_w / negative.ppi * 25.4:.0f} × {full_h / negative.ppi * 25.4:.0f} mm"
        kind = "predicted print" if self.preview_mode.get() == "proof" else "film, ink side down"
        self.preview_caption.config(text=f"{full_w} × {full_h} px{mm}   ·   {kind}")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "done":
                    output, negative = payload
                    self.process_button.config(state=NORMAL)
                    self.status.config(text=f"Saved {output}")
                    self._show_preview(negative)
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
                elif kind == "grid_result":
                    self._grid_analysis = payload
                    self._show_grid_result(payload)
                elif kind == "wedge_result":
                    self._wedge_analysis = payload
                    self._show_wedge_result(payload)
                elif kind == "batch_progress":
                    i, total, name = payload
                    self.batch_progress["maximum"] = total
                    self.batch_progress["value"] = i
                    self.batch_status.config(text=f"{i + 1} of {total}")
                    self._log_batch(f"[{i + 1}/{total}] {name}")
                elif kind == "batch_done":
                    self.batch_button.config(state=NORMAL)
                    self.batch_progress["value"] = self.batch_progress["maximum"]
                    self.batch_status.config(text="Finished.")
                    self._log_batch(payload.summary())
                elif kind == "batch_error":
                    self.batch_button.config(state=NORMAL)
                    self.batch_status.config(text="")
                    messagebox.showerror("Batch", str(payload))
                elif kind == "cal_error":
                    self.cal_status.config(text="")
                    messagebox.showerror("Calibrate", str(payload))
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

        ttk.Label(left, text="Profiles", style="Heading.TLabel").pack(anchor=W)
        ttk.Label(left, text=str(PROFILE_DIR), style="Muted.TLabel", wraplength=380).pack(
            anchor=W, pady=(0, theme.GAP)
        )
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

        self.detail = ScrolledText(tab, width=58, state=DISABLED)
        theme.style_text(self.detail)
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


    # ------------------------------------------------------------------ Batch tab

    def _build_batch_tab(self) -> None:
        tab = self.batch_tab
        ttk.Label(tab, text="Process a whole folder", style="Heading.TLabel").pack(anchor=W)
        ttk.Label(
            tab,
            text="One profile applied to every image. A file that cannot be read is reported and skipped —"
            " the rest of the run continues.",
            style="Muted.TLabel",
            wraplength=900,
        ).pack(anchor=W, pady=(theme.ROW_GAP, theme.GAP))

        box = ttk.LabelFrame(tab, text="Settings", padding=theme.GROUP_PAD)
        box.pack(fill=X)
        box.columnconfigure(1, weight=1)

        holder = self._form_row(box, "Positives folder", 0)
        self.batch_src_var = StringVar()
        ttk.Entry(holder, textvariable=self.batch_src_var, width=48).pack(side=LEFT)
        ttk.Button(
            holder, text="…", width=3, style="Small.TButton", command=lambda: self._pick_dir(self.batch_src_var)
        ).pack(side=LEFT, padx=(theme.GAP, 0))

        holder = self._form_row(box, "Output folder", 1)
        self.batch_out_var = StringVar(value="out")
        ttk.Entry(holder, textvariable=self.batch_out_var, width=48).pack(side=LEFT)
        ttk.Button(
            holder, text="…", width=3, style="Small.TButton", command=lambda: self._pick_dir(self.batch_out_var)
        ).pack(side=LEFT, padx=(theme.GAP, 0))

        holder = self._form_row(box, "Profile", 2)
        self.batch_profile_var = StringVar()
        self.batch_profile_box = ttk.Combobox(
            holder, textvariable=self.batch_profile_var, state="readonly", width=30
        )
        self.batch_profile_box.pack(side=LEFT)

        holder = self._form_row(box, "Print size (mm)", 3)
        self.batch_w_var, self.batch_h_var = StringVar(value="240"), StringVar(value="180")
        ttk.Entry(holder, textvariable=self.batch_w_var, width=theme.NARROW_WIDTH).pack(side=LEFT)
        ttk.Label(holder, text="×").pack(side=LEFT, padx=theme.GAP)
        ttk.Entry(holder, textvariable=self.batch_h_var, width=theme.NARROW_WIDTH).pack(side=LEFT)
        ttk.Label(holder, text="at").pack(side=LEFT, padx=theme.GAP)
        self.batch_ppi_var = StringVar(value=str(DEFAULT_OUTPUT_PPI))
        ttk.Entry(holder, textvariable=self.batch_ppi_var, width=6).pack(side=LEFT)
        ttk.Label(holder, text="ppi", style="Muted.TLabel").pack(side=LEFT, padx=(theme.GAP, 0))

        controls = ttk.Frame(tab)
        controls.pack(fill=X, pady=theme.WIDE_GAP)
        self.batch_button = ttk.Button(
            controls, text="Process folder", style="Accent.TButton", command=self._run_batch
        )
        self.batch_button.pack(side=LEFT)
        self.batch_progress = ttk.Progressbar(controls, mode="determinate", length=380)
        self.batch_progress.pack(side=LEFT, padx=theme.WIDE_GAP)
        self.batch_status = ttk.Label(controls, text="", style="OK.TLabel")
        self.batch_status.pack(side=LEFT)

        self.batch_log = ScrolledText(tab, height=16, state=DISABLED)
        theme.style_text(self.batch_log)
        self.batch_log.pack(fill=BOTH, expand=True)

    def _pick_dir(self, var: StringVar) -> None:
        path = filedialog.askdirectory(title="Choose folder")
        if path:
            var.set(path)

    def _log_batch(self, text: str) -> None:
        self.batch_log.config(state=NORMAL)
        self.batch_log.insert(END, text + "\n")
        self.batch_log.see(END)
        self.batch_log.config(state=DISABLED)

    def _run_batch(self) -> None:
        source = self.batch_src_var.get().strip()
        output = self.batch_out_var.get().strip()
        profile = self.profiles.get(self.batch_profile_var.get())
        if not source or not output or profile is None:
            messagebox.showinfo("Batch", "Choose a source folder, an output folder and a profile.")
            return
        if not profile.is_ready_to_print:
            messagebox.showerror(
                "Batch",
                f"Profile {profile.name!r} has no blocker colour yet — calibrate it first.",
            )
            return
        try:
            width, height = float(self.batch_w_var.get()), float(self.batch_h_var.get())
            ppi = float(self.batch_ppi_var.get())
        except ValueError as e:
            messagebox.showerror("Batch", f"Check the numeric fields: {e}")
            return

        self.batch_button.config(state=DISABLED)
        self.batch_progress["value"] = 0
        self.batch_status.config(text="Working…")
        self._log_batch(f"--- {profile.name} → {output}")
        if profile.provisional:
            self._log_batch("    note: profile is provisional — tones are a starting point")

        def work() -> None:
            def progress(i: int, total: int, path: Path) -> None:
                self.queue.put(("batch_progress", (i, total, path.name)))

            try:
                result = batch_negatives(
                    source,
                    profile,
                    PrintSize(width, height),
                    output,
                    progress=progress,
                    output_ppi=ppi,
                )
                self.queue.put(("batch_done", result))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("batch_error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ Calibrate tab

    def _build_calibrate_tab(self) -> None:
        tab = self.calibrate_tab
        self._grid_analysis: GridAnalysis | None = None
        self._wedge_analysis: WedgeAnalysis | None = None

        top = ttk.Frame(tab)
        top.pack(fill=X)
        ttk.Label(top, text="Calibrating", style="Heading.TLabel").pack(side=LEFT)
        self.cal_profile_var = StringVar()
        self.cal_profile_box = ttk.Combobox(top, textvariable=self.cal_profile_var, state="readonly", width=30)
        self.cal_profile_box.pack(side=LEFT, padx=theme.GAP)
        self.cal_profile_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_wizard_state())
        self.cal_status = ttk.Label(top, text="", style="OK.TLabel")
        self.cal_status.pack(side=LEFT, padx=theme.WIDE_GAP)

        ttk.Label(
            tab,
            text="Each step prints through the linear profile, so measurements capture the process itself"
            " rather than the process plus a curve.",
            style="Muted.TLabel",
            wraplength=1000,
        ).pack(anchor=W, pady=(theme.ROW_GAP, theme.GAP))

        # --- Step 1: exposure -------------------------------------------------
        step1, body = self._wizard_step(tab, 1, "Exposure", "Find the standard printing exposure")
        self.step1_state = step1
        ttk.Label(
            body,
            text="Print the strip, expose progressively (cover one zone per interval), process, dry."
            " SPE is the shortest time whose clear half matches the next zone's.",
            style="Muted.TLabel",
            wraplength=860,
        ).pack(anchor=W)
        row1 = ttk.Frame(body)
        row1.pack(anchor=W, pady=(theme.GAP, 0))
        ttk.Button(
            row1, text="Generate strip", style="Small.TButton", command=lambda: self._generate_target("exposure")
        ).pack(side=LEFT)
        ttk.Label(row1, text="SPE seconds").pack(side=LEFT, padx=(theme.WIDE_GAP, theme.GAP))
        self.spe_var = StringVar()
        ttk.Entry(row1, textvariable=self.spe_var, width=theme.NARROW_WIDTH).pack(side=LEFT)
        ttk.Label(row1, text="UV source").pack(side=LEFT, padx=(theme.WIDE_GAP, theme.GAP))
        self.uv_var = StringVar()
        ttk.Entry(row1, textvariable=self.uv_var, width=26).pack(side=LEFT)
        ttk.Button(row1, text="Save", style="Small.TButton", command=self._save_exposure).pack(
            side=LEFT, padx=(theme.WIDE_GAP, 0)
        )

        # --- Step 2: blocker --------------------------------------------------
        step2, body = self._wizard_step(tab, 2, "Blocker", "Measure the best UV-blocking hue")
        self.step2_state = step2
        row = ttk.Frame(body)
        row.pack(fill=X)
        ttk.Button(
            row, text="Generate grid", style="Small.TButton", command=lambda: self._generate_target("grid")
        ).pack(side=LEFT)
        ttk.Label(row, text="Scan of grid print").pack(side=LEFT, padx=(theme.WIDE_GAP, theme.GAP))
        self.grid_scan_var = StringVar()
        ttk.Entry(row, textvariable=self.grid_scan_var, width=theme.FIELD_WIDTH).pack(side=LEFT)
        ttk.Button(
            row, text="…", width=3, style="Small.TButton", command=lambda: self._pick_scan(self.grid_scan_var)
        ).pack(side=LEFT, padx=theme.GAP)
        ttk.Button(row, text="Analyse", style="Small.TButton", command=self._analyse_grid).pack(side=LEFT)

        self.grid_result = ttk.Label(body, text="Not analysed yet.", justify=LEFT, style="Mono.TLabel")
        self.grid_result.pack(anchor=W, pady=theme.GAP)

        apply_row = ttk.Frame(body)
        apply_row.pack(anchor=W)
        ttk.Label(apply_row, text="Hue°").pack(side=LEFT)
        self.hue_var = StringVar()
        ttk.Entry(apply_row, textvariable=self.hue_var, width=6).pack(side=LEFT, padx=theme.GAP)
        ttk.Label(apply_row, text="Saturation").pack(side=LEFT, padx=(theme.GAP, 0))
        self.sat_var = StringVar()
        ttk.Entry(apply_row, textvariable=self.sat_var, width=6).pack(side=LEFT, padx=theme.GAP)
        ttk.Button(apply_row, text="Save blocker", style="Small.TButton", command=self._save_blocker).pack(
            side=LEFT, padx=(theme.GAP, 0)
        )

        # --- Step 3: linearise ------------------------------------------------
        step3, body = self._wizard_step(tab, 3, "Linearisation", "Derive the correction curve", expand=True)
        self.step3_state = step3
        row = ttk.Frame(body)
        row.pack(fill=X)
        ttk.Button(
            row, text="Generate wedge", style="Small.TButton", command=lambda: self._generate_target("wedge")
        ).pack(side=LEFT)
        ttk.Label(row, text="Scan of wedge print").pack(side=LEFT, padx=(theme.WIDE_GAP, theme.GAP))
        self.wedge_scan_var = StringVar()
        ttk.Entry(row, textvariable=self.wedge_scan_var, width=theme.FIELD_WIDTH).pack(side=LEFT)
        ttk.Button(
            row, text="…", width=3, style="Small.TButton", command=lambda: self._pick_scan(self.wedge_scan_var)
        ).pack(side=LEFT, padx=theme.GAP)
        ttk.Button(row, text="Analyse", style="Small.TButton", command=self._analyse_wedge).pack(side=LEFT)

        lower = ttk.Frame(body)
        lower.pack(fill=BOTH, expand=True, pady=(theme.GAP, 0))

        from tkinter import Canvas

        self.curve_canvas = Canvas(lower, width=theme.CURVE_SIZE, height=theme.CURVE_SIZE)
        theme.style_canvas(self.curve_canvas)
        self.curve_canvas.pack(side=RIGHT, padx=(theme.WIDE_GAP, 0))

        self.wedge_result = ttk.Label(lower, text="Not analysed yet.", justify=LEFT, style="Mono.TLabel")
        self.wedge_result.pack(anchor="nw")
        self.save_measured_button = ttk.Button(
            lower,
            text="Save measured profile  (+ .cube / .acv)",
            style="Accent.TButton",
            command=self._save_measured,
            state=DISABLED,
        )
        self.save_measured_button.pack(anchor=W, pady=(theme.GAP, 0))

    def _wizard_step(self, parent, number: int, title: str, subtitle: str, expand: bool = False):
        """A numbered step panel. Returns (state_label, body_frame)."""
        box = ttk.Frame(parent)
        box.pack(fill=BOTH if expand else X, expand=expand, pady=(0, theme.GAP))

        head = ttk.Frame(box)
        head.pack(fill=X)
        ttk.Label(head, text=str(number), style="Step.TLabel", width=3).pack(side=LEFT)
        # The state label is packed before the expanding title block: pack gives space in
        # order, so a preceding expand=True sibling would leave it nothing and unmapped.
        state = ttk.Label(head, text="pending", style="Muted.TLabel", anchor="e")
        state.pack(side=RIGHT, padx=(theme.GAP, 0))
        titles = ttk.Frame(head)
        titles.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(titles, text=title, style="Heading.TLabel").pack(anchor=W)
        ttk.Label(titles, text=subtitle, style="Muted.TLabel").pack(anchor=W)

        body = ttk.Frame(box, padding=(theme.WIDE_GAP + theme.GROUP_PAD, theme.GAP, 0, 0))
        body.pack(fill=BOTH if expand else X, expand=expand)
        ttk.Separator(box, orient="horizontal").pack(fill=X, pady=(theme.GAP, 0))
        return state, body

    def _refresh_wizard_state(self) -> None:
        """Show which calibration steps this profile has actually recorded."""
        profile = self.profiles.get(self.cal_profile_var.get())
        if profile is None:
            return
        done = ("done", "OK.TLabel")
        pending = ("pending", "Muted.TLabel")

        text, style = done if profile.exposure.get("spe_seconds") else pending
        self.step1_state.config(text=text, style=style)

        text, style = done if profile.is_ready_to_print else pending
        self.step2_state.config(text=text, style=style)

        text, style = done if not profile.lut.is_identity() else pending
        self.step3_state.config(text=text, style=style)

    def _pick_scan(self, var: StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Choose scan", filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All", "*.*")]
        )
        if path:
            var.set(path)

    def _cal_profile(self) -> Profile | None:
        profile = self.profiles.get(self.cal_profile_var.get())
        if profile is None:
            messagebox.showinfo("Calibrate", "Choose a profile to calibrate first.")
        return profile

    def _save_cal_profile(self, profile: Profile) -> None:
        path = PROFILE_DIR / f"{self.cal_profile_var.get()}.json"
        profile.save(path)
        self._reload_profiles()
        self._refresh_wizard_state()
        self.cal_status.config(text=f"Saved {path.name}")

    def _save_exposure(self) -> None:
        profile = self._cal_profile()
        if profile is None:
            return
        try:
            spe = float(self.spe_var.get())
        except ValueError:
            messagebox.showerror("Calibrate", "SPE must be a number of seconds.")
            return
        profile.exposure = {"spe_seconds": spe, "uv_source": self.uv_var.get().strip()}
        self._save_cal_profile(profile)

    def _sidecar_for(self, kind: str) -> Path | None:
        path = Path("targets") / f"{kind}.json"
        if not path.exists():
            messagebox.showerror(
                "Calibrate",
                f"No sidecar at {path}. Generate the target here first — the sidecar written "
                "at generation time is what maps the scan back to patch values.",
            )
            return None
        return path

    def _analyse_grid(self) -> None:
        scan = self.grid_scan_var.get().strip()
        sidecar = self._sidecar_for("blocker_grid")
        if not scan or sidecar is None:
            if not scan:
                messagebox.showinfo("Calibrate", "Choose the scan of the grid print.")
            return
        self.cal_status.config(text="Analysing grid…")

        def work() -> None:
            try:
                self.queue.put(("grid_result", analyze_grid(scan, sidecar)))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("cal_error", f"Grid analysis failed: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def _show_grid_result(self, result: GridAnalysis) -> None:
        self.cal_status.config(text="Grid analysed.")
        self.grid_result.config(text=result.summary())
        self.hue_var.set(f"{result.best['hue_deg']:g}")
        if result.recommended_saturation is not None:
            self.sat_var.set(f"{result.recommended_saturation:g}")

    def _save_blocker(self) -> None:
        profile = self._cal_profile()
        if profile is None:
            return
        try:
            hue = float(self.hue_var.get())
            sat = float(self.sat_var.get())
        except ValueError:
            messagebox.showerror("Calibrate", "Enter numeric hue and saturation (run Analyse first).")
            return
        profile.blocker = {
            "model": "fixed_hue",
            "rgb": list(hue_to_rgb(hue)),
            "saturation": sat,
        }
        self._save_cal_profile(profile)

    def _analyse_wedge(self) -> None:
        scan = self.wedge_scan_var.get().strip()
        sidecar = self._sidecar_for("step_wedge")
        if not scan or sidecar is None:
            if not scan:
                messagebox.showinfo("Calibrate", "Choose the scan of the wedge print.")
            return
        self.cal_status.config(text="Analysing wedge…")

        def work() -> None:
            try:
                self.queue.put(("wedge_result", analyze_wedge(scan, sidecar)))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("cal_error", f"Wedge analysis failed: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def _show_wedge_result(self, result: WedgeAnalysis) -> None:
        self.cal_status.config(text="Wedge analysed.")
        self.wedge_result.config(text=result.summary())
        self.save_measured_button.config(state=NORMAL)
        # Draw the derived correction curve.
        c = self.curve_canvas
        c.delete("all")
        size = theme.CURVE_SIZE
        c.create_line(0, size, size, 0, fill=theme.CURVE_REFERENCE, dash=(3, 3))  # identity reference
        points = []
        for i, v in enumerate(result.lut.values):
            x = i / (len(result.lut.values) - 1) * (size - 1)
            y = (1.0 - v) * (size - 1)
            points.extend((x, y))
        c.create_line(*points, fill=theme.CURVE, width=2, smooth=True)

    def _save_measured(self) -> None:
        profile = self._cal_profile()
        result = self._wedge_analysis
        if profile is None or result is None:
            return
        import datetime

        profile.lut = result.lut
        profile.provisional = False
        profile.measurements = {
            "raw_patches": result.raw_patches,
            "scan_date": datetime.date.today().isoformat(),
            "density_range": round(result.density_range, 3),
            "spikes": result.spikes,
        }
        self._save_cal_profile(profile)
        stem = PROFILE_DIR / self.cal_profile_var.get()
        result.lut.export_cube(stem.with_suffix(".cube"))
        result.lut.export_acv(stem.with_suffix(".acv"))
        messagebox.showinfo(
            "Calibrate",
            f"Measured profile saved.\nExported {stem.with_suffix('.cube').name} and "
            f"{stem.with_suffix('.acv').name} for Photoshop QA.",
        )


def main() -> int:
    root = Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

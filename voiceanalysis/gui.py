from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .pipeline import analyze_pair


class VoiceAnalysisApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Voice Analysis Pipeline")
        self.geometry("620x300")
        self.file_a = tk.StringVar()
        self.file_b = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "analysis_output"))
        self.status = tk.StringVar(value="Choose two WAV files, then press Analyze.")
        self.skip_embeddings = tk.BooleanVar(value=False)
        self._build()

    def _build(self) -> None:
        pad = {"padx": 12, "pady": 6}
        for row, (label, var) in enumerate((("Recording A", self.file_a), ("Recording B", self.file_b), ("Output folder", self.output_dir))):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", **pad)
            ttk.Entry(self, textvariable=var, width=58).grid(row=row, column=1, **pad)
            ttk.Button(self, text="Browse", command=lambda v=var, r=row: self._browse(v, r)).grid(row=row, column=2, **pad)
        ttk.Checkbutton(self, text="Skip neural embeddings", variable=self.skip_embeddings).grid(row=3, column=1, sticky="w", **pad)
        ttk.Button(self, text="Analyze", command=self._run).grid(row=4, column=1, **pad)
        ttk.Label(self, textvariable=self.status, wraplength=560).grid(row=5, column=0, columnspan=3, sticky="w", **pad)

    def _browse(self, var: tk.StringVar, row: int) -> None:
        path = filedialog.askdirectory() if row == 2 else filedialog.askopenfilename(filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _run(self) -> None:
        def worker() -> None:
            try:
                self.status.set("Analyzing locally; this may take several minutes...")
                pdf = analyze_pair(self.file_a.get(), self.file_b.get(), self.output_dir.get(), self.skip_embeddings.get())
                self.status.set(f"Done: {pdf}")
                messagebox.showinfo("Voice Analysis", f"Report written to:\n{pdf}")
            except Exception as exc:
                self.status.set(f"Analysis failed: {exc}")
                messagebox.showerror("Voice Analysis", str(exc))
        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    VoiceAnalysisApp().mainloop()


if __name__ == "__main__":
    main()

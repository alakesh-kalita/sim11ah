import math
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sim11ah.Results.plot_simulator_results import ResultsProcessor

class StatCard(tk.Frame):
    def __init__(self, parent, title: str, accent="#4f46e5", bg="#ffffff", fg="#111827"):
        super().__init__(parent, bd=0, highlightthickness=0, bg=bg)
        self.configure(padx=12, pady=10)

        self.topbar = tk.Frame(self, bg=accent, height=5)
        self.topbar.pack(fill="x", side="top")

        body = tk.Frame(self, bg=bg)
        body.pack(fill="both", expand=True, pady=(8, 0))

        self.title = tk.Label(body, text=title, font=("Arial", 11, "bold"), bg=bg, fg=fg)
        self.big = tk.Label(body, text="-", font=("Arial", 20, "bold"), bg=bg, fg=fg)
        self.sub = tk.Label(body, text="", font=("Arial", 10), bg=bg, fg="#4b5563")

        self.title.pack(anchor="w")
        self.big.pack(anchor="w", pady=(6, 2))
        self.sub.pack(anchor="w")

    def set(self, big: str, sub: str = ""):
        self.big.config(text=big)
        self.sub.config(text=sub)


class NodeWiseBarChart(tk.Canvas):
    def __init__(self, parent, title="Chart", width=520, height=240, bg="#ffffff"):
        super().__init__(
            parent,
            width=width,
            height=height,
            bd=0,
            highlightthickness=1,
            highlightbackground="#d1d5db",
            bg=bg,
            xscrollincrement=1
        )
        self.w = width
        self.h = height
        self.title = title
        self.palette = [
            "#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
            "#14b8a6", "#f97316", "#ec4899", "#84cc16", "#06b6d4"
        ]
        self.config(scrollregion=(0, 0, self.w, self.h))

    def draw(self, data_dict):
        self.delete("all")

        self.create_text(
            14, 14,
            anchor="nw",
            text=self.title,
            font=("Arial", 11, "bold"),
            fill="#111827"
        )

        if not data_dict:
            self.config(scrollregion=(0, 0, self.w, self.h))
            self.create_text(
                self.w / 2, self.h / 2,
                text="No data yet",
                font=("Arial", 11),
                fill="#6b7280"
            )
            return

        items = sorted(data_dict.items(), key=lambda x: x[0])
        n = len(items)

        left = 55
        top = 52
        bottom = self.h - 35
        gap = 8
        bar_w = 18
        right_padding = 20

        total_w = max(self.w, left + right_padding + n * (bar_w + gap) + gap)
        right = total_w - right_padding

        self.config(scrollregion=(0, 0, total_w, self.h))

        chart_h = max(1, bottom - top)
        max_val = max(v for _, v in items)
        if max_val <= 0:
            max_val = 1

        self.create_line(left, top, left, bottom, fill="#6b7280", width=1)
        self.create_line(left, bottom, right, bottom, fill="#6b7280", width=1)

        y_ticks = 5
        for i in range(y_ticks + 1):
            y = bottom - (i / y_ticks) * chart_h
            val = (i / y_ticks) * max_val
            self.create_line(left - 4, y, left, y, fill="#6b7280")
            self.create_text(left - 8, y, text=str(int(val)), anchor="e", font=("Arial", 8), fill="#4b5563")
            if i < y_ticks:
                self.create_line(left, y, right, y, fill="#eef2f7", dash=(2, 4))

        value_label_stride = max(1, math.ceil(n / 25))
        x_label_stride = max(1, math.ceil(n / 30))

        for idx, (node_id, value) in enumerate(items):
            x0 = left + gap + idx * (bar_w + gap)
            x1 = x0 + bar_w

            bar_h = (value / max_val) * chart_h
            y0 = bottom - bar_h
            y1 = bottom

            color = self.palette[idx % len(self.palette)]

            self.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

            show_value = (n <= 20) or (idx % value_label_stride == 0) or (idx == n - 1)
            if show_value:
                self.create_text(
                    (x0 + x1) / 2,
                    max(top - 8, y0 - 8),
                    text=str(value),
                    font=("Arial", 8, "bold"),
                    fill="#111827"
                )

            show_x_label = (n <= 25) or (idx % x_label_stride == 0) or (idx == n - 1)
            if show_x_label:
                self.create_text(
                    (x0 + x1) / 2,
                    bottom + 12,
                    text=str(node_id),
                    font=("Arial", 8),
                    fill="#374151"
                )


class Dashboard(tk.Tk):
    def __init__(self, sim, sim_builder=None, initial_settings=None):
        super().__init__()
        self.sim = sim
        self.sim_builder = sim_builder
        self.running = False
        self._applied_signature = None

        self.bg_main = "#eef4ff"
        self.bg_panel = "#ffffff"
        self.text_main = "#0f172a"

        self.title("WiFi HaLow IEEE 802.11ah")
        self.geometry("1360x880")
        self.minsize(1120, 740)
        self.configure(bg=self.bg_main)

        self._setup_style()

        settings = initial_settings or {}
        self.sel_num_stas = tk.IntVar(value=int(settings.get("num_stas", 50)))
        self.sel_seed = tk.IntVar(value=int(settings.get("seed", 0)))
        self.sel_traffic = tk.StringVar(value=str(settings.get("traffic", "periodic")))
        self.sel_raw = tk.StringVar(value="Enabled" if bool(settings.get("raw_enable", True)) else "Disabled")

        self._applied_signature = self._current_settings_signature()

        cards = tk.Frame(self, bg=self.bg_main)
        cards.pack(fill="x", padx=12, pady=12)

        self.card_pdr = StatCard(cards, "Overall E2E PDR", accent="#2563eb", bg=self.bg_panel)
        self.card_lat = StatCard(cards, "Overall E2E Latency", accent="#10b981", bg=self.bg_panel)
        self.card_thr = StatCard(cards, "Throughput", accent="#f59e0b", bg=self.bg_panel)

        for c in (self.card_pdr, self.card_lat, self.card_thr):
            c.pack(side="left", padx=7, fill="x", expand=True)

        mid = tk.Frame(self, bg=self.bg_main)
        mid.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left = tk.Frame(mid, bg=self.bg_main)
        left.pack(side="left", fill="both", expand=True)

        right = tk.Frame(mid, width=260, bg=self.bg_panel, bd=0, highlightthickness=1, highlightbackground="#d1d5db")
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        chart_frame = tk.Frame(left, bg=self.bg_main)
        chart_frame.pack(fill="x", pady=(0, 10))

        gen_wrap = tk.Frame(chart_frame, bg=self.bg_panel, bd=0, highlightthickness=1, highlightbackground="#d1d5db")
        gen_wrap.pack(fill="x", pady=(0, 10))

        self.gen_chart = NodeWiseBarChart(
            gen_wrap,
            title="Node-wise Packets Generated",
            width=900,
            height=210,
            bg=self.bg_panel
        )
        self.gen_chart.pack(fill="x", side="top")

        self.gen_xscroll = tk.Scrollbar(gen_wrap, orient="horizontal", command=self.gen_chart.xview)
        self.gen_xscroll.pack(fill="x", side="bottom")
        self.gen_chart.configure(xscrollcommand=self.gen_xscroll.set)

        del_wrap = tk.Frame(chart_frame, bg=self.bg_panel, bd=0, highlightthickness=1, highlightbackground="#d1d5db")
        del_wrap.pack(fill="x")

        self.del_chart = NodeWiseBarChart(
            del_wrap,
            title="Node-wise Packets Delivered",
            width=900,
            height=210,
            bg=self.bg_panel
        )
        self.del_chart.pack(fill="x", side="top")

        self.del_xscroll = tk.Scrollbar(del_wrap, orient="horizontal", command=self.del_chart.xview)
        self.del_xscroll.pack(fill="x", side="bottom")
        self.del_chart.configure(xscrollcommand=self.del_xscroll.set)

        log_wrap = tk.Frame(left, bg=self.bg_panel, bd=0, highlightthickness=1, highlightbackground="#d1d5db")
        log_wrap.pack(fill="both", expand=True, pady=(10, 0))

        tk.Label(
            log_wrap,
            text="Log Viewer (tail)",
            font=("Arial", 12, "bold"),
            bg=self.bg_panel,
            fg=self.text_main
        ).pack(anchor="w", padx=10, pady=(10, 6))

        log_text_wrap = tk.Frame(log_wrap, bg=self.bg_panel)
        log_text_wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(
            log_text_wrap,
            height=40,
            bg="#0b1220",
            fg="#e5e7eb",
            insertbackground="#ffffff",
            relief="flat",
            font=("Courier New", 10),
            wrap="none"
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        self.log_scroll_y = tk.Scrollbar(log_text_wrap, orient="vertical", command=self.log_text.yview)
        self.log_scroll_y.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=self.log_scroll_y.set)
        self.log_text.insert("end", "Press START to begin simulation...\n")

        tk.Label(
            right,
            text="Controls",
            font=("Arial", 13, "bold"),
            bg=self.bg_panel,
            fg=self.text_main
        ).pack(pady=(12, 12), anchor="w", padx=12)

        self.update_ms = 150
        self.step_dt = 0.2

        control_row = tk.Frame(right, bg=self.bg_panel)
        control_row.pack(fill="x", padx=12, pady=4)

        ttk.Button(control_row, text="Start", command=self.start).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(control_row, text="Stop", command=self.stop).pack(side="left", expand=True, fill="x", padx=(4, 0))

        ttk.Button(right, text="Step", command=self.step_once).pack(fill="x", padx=12, pady=4)

        results_row = tk.Frame(right, bg=self.bg_panel)
        results_row.pack(fill="x", padx=12, pady=10)

        ttk.Button(results_row, text="Export Logs CSV", command=self.export_csv).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        ttk.Button(results_row, text="Results", command=self.run_results).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=12, pady=14)

        tk.Label(
            right,
            text="Simulation Settings",
            font=("Arial", 12, "bold"),
            bg=self.bg_panel,
            fg=self.text_main
        ).pack(anchor="w", padx=12, pady=(0, 10))

        tk.Label(right, text="Number of STAs", bg=self.bg_panel, fg=self.text_main, font=("Arial", 10, "bold")).pack(anchor="w", padx=12)
        self.cbo_num_stas = ttk.Combobox(
            right,
            textvariable=self.sel_num_stas,
            values=[10, 20, 50, 100, 150, 200, 300, 500],
            state="readonly"
        )
        self.cbo_num_stas.pack(fill="x", padx=12, pady=(2, 8))

        tk.Label(right, text="Traffic Model", bg=self.bg_panel, fg=self.text_main, font=("Arial", 10, "bold")).pack(anchor="w", padx=12)
        self.cbo_traffic = ttk.Combobox(
            right,
            textvariable=self.sel_traffic,
            values=["periodic", "poisson", "cbr", "bursty", "onoff"],
            state="readonly"
        )
        self.cbo_traffic.pack(fill="x", padx=12, pady=(2, 8))

        tk.Label(right, text="RAW", bg=self.bg_panel, fg=self.text_main, font=("Arial", 10, "bold")).pack(anchor="w", padx=12)
        self.cbo_raw = ttk.Combobox(
            right,
            textvariable=self.sel_raw,
            values=["Enabled", "Disabled"],
            state="readonly"
        )
        self.cbo_raw.pack(fill="x", padx=12, pady=(2, 8))

        tk.Label(right, text="Seed", bg=self.bg_panel, fg=self.text_main, font=("Arial", 10, "bold")).pack(anchor="w", padx=12)
        self.cbo_seed = ttk.Combobox(
            right,
            textvariable=self.sel_seed,
            values=[0, 1, 2, 3, 4, 5, 10, 20, 42, 100],
            state="readonly"
        )
        self.cbo_seed.pack(fill="x", padx=12, pady=(2, 10))

        ttk.Button(right, text="Apply Settings", command=self.apply_settings).pack(fill="x", padx=12, pady=(4, 10))

        tk.Label(
            right,
            text="WiFi HaLow Simulator\nDeveloped by WindS Lab",
            bg=self.bg_panel,
            fg="#475569",
            font=("Arial", 9, "bold"),
            justify="center"
        ).pack(fill="x", padx=12, pady=(2, 8))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=12, pady=10)

        self.status = tk.StringVar(value="Ready. Press START to begin simulation.")
        tk.Label(
            right,
            textvariable=self.status,
            wraplength=230,
            justify="left",
            bg=self.bg_panel,
            fg="#334155",
            font=("Arial", 10)
        ).pack(anchor="w", padx=12)

        self.after(self.update_ms, self._tick)

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", font=("Arial", 10, "bold"), padding=8)
        style.configure("TSpinbox", padding=5)
        style.configure("TCombobox", padding=4)

    def _current_settings_signature(self):
        return (
            int(self.sel_num_stas.get()),
            int(self.sel_seed.get()),
            str(self.sel_traffic.get()).lower().strip(),
            str(self.sel_raw.get()).lower().startswith("enabled"),
        )

    def _shutdown_current_sim(self):
        try:
            for n in self.sim.nodes.values():
                try:
                    n.stop()
                except Exception:
                    pass
            for n in self.sim.nodes.values():
                try:
                    n.finalize()
                except Exception:
                    pass
        except Exception:
            pass

    def _rebuild_sim_from_gui(self):
        if self.sim_builder is None:
            raise RuntimeError("Simulator builder not provided.")

        num_stas, seed, traffic, raw_enable = self._current_settings_signature()

        self._shutdown_current_sim()
        self.sim = self.sim_builder(
            num_stas=num_stas,
            seed=seed,
            traffic=traffic,
            raw_enable=raw_enable
        )
        self._applied_signature = (num_stas, seed, traffic, raw_enable)

    def apply_settings(self):
        was_running = self.running
        self.running = False

        try:
            self._rebuild_sim_from_gui()

            self.log_text.delete("1.0", "end")
            self.log_text.insert("end", "Settings applied. Press START to begin simulation...\n")

            self.card_pdr.set("-", "")
            self.card_lat.set("-", "")
            self.card_thr.set("-", "")

            self.gen_chart.draw({})
            self.del_chart.draw({})

            num_stas, seed, traffic, raw_enable = self._applied_signature
            self.status.set(
                f"Settings applied: STAs={num_stas}, traffic={traffic}, RAW={'on' if raw_enable else 'off'}, seed={seed}"
            )

            if was_running:
                self.running = True
                self._refresh()

        except Exception as e:
            self.status.set(f"Apply settings error: {e}")
            raise

    def start(self):
        try:
            current_signature = self._current_settings_signature()
            if current_signature != self._applied_signature:
                self._rebuild_sim_from_gui()

            if not self.running:
                self.running = True
                self.status.set("Running...")
                self._refresh()
        except Exception as e:
            self.status.set(f"Start error: {e}")
            raise

    def stop(self):
        self.running = False
        self.status.set("Stopped.")

    def step_once(self):
        try:
            current_signature = self._current_settings_signature()
            if current_signature != self._applied_signature:
                self._rebuild_sim_from_gui()

            self.sim.step(self.step_dt)
            self._refresh()
        except Exception as e:
            self.status.set(f"Step error: {e}")
            raise

    def _tick(self):
        try:
            if self.running:
                self.sim.step(self.step_dt)
                self._refresh()
        except Exception as e:
            self.running = False
            self.status.set(f"Tick error: {e}")
            raise

        self.after(self.update_ms, self._tick)

    def _compute_metrics_from_logs(self):
        logs = list(self.sim.logger.logs)

        tx = 0
        rx = 0
        drop = 0
        delays = []

        node_generated = {}
        node_delivered = {}

        for r in logs:
            layer = str(r.get("layer", ""))
            event = str(r.get("event", ""))
            details = r.get("details", {}) or {}
            node_id = r.get("node_id")

            if layer == "APP" and event == "GENERATE":
                tx += 1
                if node_id is not None:
                    node_generated[node_id] = node_generated.get(node_id, 0) + 1

            elif layer == "APP" and event == "DELIVER":
                rx += 1
                dst = None

                pkt = r.get("packet", None)
                if isinstance(pkt, dict):
                    dst = pkt.get("dst", None)

                if dst is None:
                    dst = details.get("dst", None)

                if dst is None:
                    dst = node_id

                if dst is not None:
                    node_delivered[dst] = node_delivered.get(dst, 0) + 1

                d = details.get("delay_s", None)
                if d is not None:
                    try:
                        delays.append(float(d))
                    except Exception:
                        pass

            elif (layer == "APP" and event == "TX_FAILURE") or (layer == "MAC" and event == "DROP"):
                drop += 1

        pdr = (rx / tx) if tx > 0 else 0.0
        lat_avg = (sum(delays) / len(delays)) if delays else 0.0
        lat_max = max(delays) if delays else 0.0
        lat_min = min(delays) if delays else 0.0

        pkt_bytes = int(self.sim.config.get("app", {}).get("packet_size_bytes", 0))
        sim_time = float(getattr(self.sim.engine, "now", 0.0))
        thr_bps = (rx * pkt_bytes * 8) / max(sim_time, 1e-12)

        return {
            "time": sim_time,
            "tx": tx,
            "rx": rx,
            "drop": drop,
            "pdr": pdr,
            "lat_avg": lat_avg,
            "lat_max": lat_max,
            "lat_min": lat_min,
            "thr_bps": thr_bps,
            "node_generated": node_generated,
            "node_delivered": node_delivered,
        }

    def _refresh(self):
        try:
            m = self._compute_metrics_from_logs()

            sim_time = float(m["time"])
            tx = int(m["tx"])
            rx = int(m["rx"])
            drop = int(m["drop"])
            pdr = float(m["pdr"])
            lat_avg = float(m["lat_avg"])
            lat_max = float(m["lat_max"])
            lat_min = float(m["lat_min"])
            thr_bps = float(m["thr_bps"])

            self.card_pdr.set(f"{pdr * 100:.2f}%", f"TX: {tx}   RX: {rx}   Drop: {drop}")
            self.card_lat.set(f"Avg {lat_avg:.2f}s", f"Max {lat_max:.2f}s   Min {lat_min:.2f}s")
            self.card_thr.set(f"{thr_bps:.1f} bps", f"Sim time: {sim_time:.2f}s")

            self.gen_chart.draw(m["node_generated"])
            self.del_chart.draw(m["node_delivered"])

            tail = self.sim.logger.dump_tail(200, human_readable=True)
            self.log_text.delete("1.0", "end")
            self.log_text.insert("end", tail)
            self.log_text.see("end")

            self.status.set(f"Sim time: {sim_time:.3f}s | Logs: {len(self.sim.logger.logs)}")
        except Exception as e:
            self.status.set(f"Refresh error: {e}")
            raise

    def export_csv(self):
        path = filedialog.asksaveasfilename(
            title="Save simulator logs",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            count = self.sim.logger.export_logs_csv(path)
            messagebox.showinfo("Export OK", f"Saved {count} log entries to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))


    def run_results(self):
        temp_csv_path = filedialog.asksaveasfilename(
            title="Save simulator logs for results processing",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not temp_csv_path:
            return

        try:
            count = self.sim.logger.export_logs_csv(temp_csv_path)
            if count <= 0:
                messagebox.showwarning("No Logs", "No log entries are available to process.")
                return

            # ✅ Import here (safe)
            from sim11ah.Results.plot_simulator_results import ResultsProcessor

            self.status.set("Processing results...")
            self.update_idletasks()

            processor = ResultsProcessor(temp_csv_path)
            processor.run()

            self.status.set("Results generated successfully.")
            messagebox.showinfo("Results", "Plots generated successfully.")

        except Exception as e:
            self.status.set(f"Results error: {e}")
            messagebox.showerror("Results Error", str(e))
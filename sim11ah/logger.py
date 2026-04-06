import csv
import json
import threading
from collections import Counter
from typing import Any, Dict, List, Optional

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    _WIDGETS_AVAILABLE = True
except ImportError:
    _WIDGETS_AVAILABLE = False


class SimLogger:
    """
    Thread-safe in-memory logger with:
      - read cursor support
      - notebook live viewer (ipywidgets) with optional auto-refresh
      - CSV export from stable snapshots
      - optional max_records cap to prevent unbounded growth
      - summary helpers for analysis
      - optional flattened CSV export of selected detail keys
    """

    _CSV_FIELDS = [
        "time", "node_id", "layer", "event",
        "packet_seq", "net_seq", "tp_seq", "frame_seq", "tx_seq",
        "ftype", "src", "dst", "next_hop",
        "details",
    ]

    # Common detail keys worth flattening for analysis
    _DEFAULT_DETAIL_EXPORT_KEYS = [
        "mode",
        "retry",
        "slot",
        "group_id",
        "config_id",
        "enter",
        "exit",
        "duration",
        "collided",
        "per_drop",
        "rx_ok",
        "drop_reason",
        "latency_s",
        "rssi_dbm",
        "noise_dbm",
        "snr_db",
        "sinr_db",
        "interference_dbm",
        "ack_for",
        "frame_seq",
        "timeout_s",
        "saved_backoff",
        "backoff_slots_left",
        "cw",
        "q_len",
        "dst",
        "src",
        "slot_exit",
        "remaining",
        "budget",
        "t_data",
        "t_ack",
        "prop",
        "sifs",
        "ack_guard",
        "beacon_count",
        "is_dtim",
        "rps_count",
    ]

    def __init__(self, max_records: int = 0):
        self._logs: List[Dict[str, Any]] = []
        self._read_cursor: int = 0
        self._lock = threading.Lock()

        self.max_records = int(max_records)

        self._out_widget: Optional[Any] = None
        self._live_controls: Optional[Any] = None
        self._live_thread: Optional[threading.Thread] = None
        self._live_stop_evt = threading.Event()

        self._live_tail_n: int = 200
        self._live_human: bool = True

    @property
    def logs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._logs)

    def log(self, rec: Dict[str, Any]) -> None:
        with self._lock:
            self._logs.append(rec)

            if self.max_records > 0 and len(self._logs) > self.max_records:
                overflow = len(self._logs) - self.max_records
                del self._logs[:overflow]
                self._read_cursor = max(0, self._read_cursor - overflow)

    def clear(self) -> None:
        with self._lock:
            self._logs.clear()
            self._read_cursor = 0

    def mark_read(self) -> None:
        with self._lock:
            self._read_cursor = len(self._logs)

    def reset_cursor(self) -> None:
        with self._lock:
            self._read_cursor = 0

    def new_entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._logs[self._read_cursor:])

    def pending_count(self) -> int:
        with self._lock:
            return len(self._logs) - self._read_cursor

    def filter_logs(self, **kwargs) -> List[Dict[str, Any]]:
        with self._lock:
            snap = list(self._logs)
        return [r for r in snap if all(r.get(k) == v for k, v in kwargs.items())]

    def filter_new_logs(self, **kwargs) -> List[Dict[str, Any]]:
        snap = self.new_entries()
        return [r for r in snap if all(r.get(k) == v for k, v in kwargs.items())]

    def filter_time_range(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        *,
        layer: Optional[str] = None,
        event: Optional[str] = None,
        node_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            snap = list(self._logs)

        out = []
        for r in snap:
            try:
                t = float(r.get("time", 0.0))
            except Exception:
                continue

            if start_time is not None and t < start_time:
                continue
            if end_time is not None and t > end_time:
                continue
            if layer is not None and r.get("layer") != layer:
                continue
            if event is not None and r.get("event") != event:
                continue
            if node_id is not None and r.get("node_id") != node_id:
                continue
            out.append(r)
        return out

    def tail_records(self, n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            snap = list(self._logs)
        return snap[-n:] if n <= len(snap) else snap

    def range_records(self, start: int, end: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            snap = list(self._logs)
        end = len(snap) if end is None else end
        return snap[start:end]

    def count_by_event(self, *, layer: Optional[str] = None) -> Dict[str, int]:
        with self._lock:
            snap = list(self._logs)
        c = Counter()
        for r in snap:
            if layer is not None and r.get("layer") != layer:
                continue
            c[str(r.get("event", ""))] += 1
        return dict(c)

    def count_by_layer(self) -> Dict[str, int]:
        with self._lock:
            snap = list(self._logs)
        c = Counter(str(r.get("layer", "")) for r in snap)
        return dict(c)

    def count_by_node(self) -> Dict[int, int]:
        with self._lock:
            snap = list(self._logs)
        c = Counter()
        for r in snap:
            try:
                c[int(r.get("node_id"))] += 1
            except Exception:
                pass
        return dict(c)

    @staticmethod
    def _safe_json(x: Any) -> str:
        try:
            return json.dumps(x, ensure_ascii=False, sort_keys=True)
        except Exception:
            return repr(x)

    @staticmethod
    def _format_record(r: Dict[str, Any]) -> str:
        try:
            t = f"{float(r.get('time', 0.0)):.6f}"
        except Exception:
            t = "0.000000"

        nid = r.get("node_id", "?")
        layer = r.get("layer", "?")
        ev = r.get("event", "?")

        ids = []
        for key, label in (
            ("packet_seq", "pkt"),
            ("net_seq", "net"),
            ("tp_seq", "tp"),
            ("frame_seq", "frm"),
            ("tx_seq", "tx"),
            ("ftype", "type"),
        ):
            if r.get(key) is not None:
                ids.append(f"{label}={r.get(key)}")

        if r.get("src") is not None:
            ids.append(f"src={r.get('src')}")
        if r.get("dst") is not None:
            ids.append(f"dst={r.get('dst')}")
        if r.get("next_hop") is not None:
            ids.append(f"nh={r.get('next_hop')}")

        details = r.get("details", {})
        id_str = (" " + " ".join(ids)) if ids else ""
        return f"[t={t}] node={nid} {layer}:{ev}{id_str} {details}"

    def _format_records(self, records: List[Dict[str, Any]]) -> str:
        if not records:
            return "(no entries)"
        return "\n".join(self._format_record(r) for r in records)

    def open_live_view(self, title: str = "Simulation Log", height: str = "420px") -> None:
        if not _WIDGETS_AVAILABLE:
            print("ipywidgets not available — falling back to plain print.")
            return

        self._out_widget = widgets.Output(
            layout=widgets.Layout(
                height=height,
                overflow_y="auto",
                border="1px solid #ccc",
                padding="6px",
            )
        )
        header = widgets.HTML(f"<b>{title}</b>")

        self._tail_slider = widgets.IntSlider(
            value=self._live_tail_n, min=20, max=2000, step=20,
            description="tail", continuous_update=False,
            layout=widgets.Layout(width="260px"),
        )
        self._human_chk = widgets.Checkbox(
            value=self._live_human, description="human",
            layout=widgets.Layout(width="120px"),
        )
        self._auto_toggle = widgets.ToggleButton(
            value=False, description="auto", button_style="",
            layout=widgets.Layout(width="90px"),
        )
        self._interval = widgets.FloatText(
            value=0.25, description="sec",
            layout=widgets.Layout(width="160px"),
        )
        self._btn_new = widgets.Button(description="show new", layout=widgets.Layout(width="110px"))
        self._btn_all = widgets.Button(description="show all", layout=widgets.Layout(width="110px"))
        self._btn_mark = widgets.Button(description="mark read", layout=widgets.Layout(width="110px"))
        self._btn_clear = widgets.Button(description="clear", layout=widgets.Layout(width="90px"))

        def _on_show_new(_):
            self._live_tail_n = int(self._tail_slider.value)
            self._live_human = bool(self._human_chk.value)
            self.show_new_logs(
                limit=None,
                advance_cursor=True,
                human_readable=self._live_human,
                tail_n=self._live_tail_n,
            )

        def _on_show_all(_):
            self._live_tail_n = int(self._tail_slider.value)
            self._live_human = bool(self._human_chk.value)
            self.show_all_logs(
                limit=None,
                human_readable=self._live_human,
                tail_n=self._live_tail_n,
            )

        def _on_mark(_):
            self.mark_read()
            self._write_to_widget("(cursor advanced)")

        def _on_clear(_):
            self.clear()
            self._write_to_widget("(log cleared)")

        self._btn_new.on_click(_on_show_new)
        self._btn_all.on_click(_on_show_all)
        self._btn_mark.on_click(_on_mark)
        self._btn_clear.on_click(_on_clear)

        controls = widgets.HBox([
            self._btn_new, self._btn_all, self._btn_mark, self._btn_clear,
            self._tail_slider, self._human_chk, self._auto_toggle, self._interval
        ])
        box = widgets.VBox([header, controls, self._out_widget])
        self._live_controls = box
        display(box)

        def _auto_changed(change):
            if change["name"] != "value":
                return
            if bool(change["new"]):
                self._start_auto_refresh()
            else:
                self._stop_auto_refresh()

        self._auto_toggle.observe(_auto_changed)

    def close_live_view(self) -> None:
        self._stop_auto_refresh()
        self._out_widget = None
        self._live_controls = None

    def _write_to_widget(self, text: str) -> None:
        if self._out_widget is None:
            print(text)
            return
        with self._out_widget:
            clear_output(wait=True)
            print(text)

    def _start_auto_refresh(self) -> None:
        if self._live_thread is not None and self._live_thread.is_alive():
            return
        self._live_stop_evt.clear()

        def _loop():
            while not self._live_stop_evt.is_set():
                try:
                    sec = float(getattr(self, "_interval").value)
                except Exception:
                    sec = 0.25

                try:
                    tail_n = int(getattr(self, "_tail_slider").value)
                except Exception:
                    tail_n = self._live_tail_n

                try:
                    human = bool(getattr(self, "_human_chk").value)
                except Exception:
                    human = self._live_human

                self.show_new_logs(
                    limit=None,
                    advance_cursor=True,
                    human_readable=human,
                    tail_n=tail_n,
                )
                self._live_stop_evt.wait(timeout=max(0.05, sec))

        self._live_thread = threading.Thread(target=_loop, daemon=True)
        self._live_thread.start()

    def _stop_auto_refresh(self) -> None:
        self._live_stop_evt.set()

    def show_new_logs(
        self,
        limit: Optional[int] = None,
        advance_cursor: bool = True,
        human_readable: bool = True,
        tail_n: Optional[int] = None,
    ) -> None:
        with self._lock:
            start = self._read_cursor
            end = len(self._logs)
            entries = list(self._logs[start:end])
            if advance_cursor:
                self._read_cursor = end

            total = end
            unread = end - start

        display_entries = entries if limit is None else entries[:limit]

        if tail_n is not None and tail_n > 0 and len(display_entries) > tail_n:
            display_entries = display_entries[-tail_n:]
            tailed = True
        else:
            tailed = False

        if not human_readable:
            text = "\n".join(str(r) for r in display_entries) or "(no new entries)"
        else:
            omitted_by_limit = (len(entries) - len(display_entries)) if (limit is not None) else 0
            header = (
                f"--- {unread} new entr{'y' if unread == 1 else 'ies'} "
                f"(total in log: {total})"
                + (f"  [limit omitted: {omitted_by_limit}]" if omitted_by_limit else "")
                + (f"  [tailed to last {tail_n}]" if tailed else "")
                + " ---"
            )
            body = self._format_records(display_entries)
            text = f"{header}\n{body}" if display_entries else "(no new log entries)"

        self._write_to_widget(text)

    def show_all_logs(
        self,
        limit: Optional[int] = None,
        human_readable: bool = True,
        tail_n: Optional[int] = None,
    ) -> None:
        with self._lock:
            snap = list(self._logs)

        records = snap if limit is None else snap[:limit]

        if tail_n is not None and tail_n > 0 and len(records) > tail_n:
            records = records[-tail_n:]
            tailed = True
        else:
            tailed = False

        if not human_readable:
            text = "\n".join(str(r) for r in records) or "(empty log)"
        else:
            count = len(records)
            total = len(snap)
            header = (
                f"--- full log: {count} entr{'y' if count == 1 else 'ies'}"
                + (f" (showing {count}/{total})" if limit else "")
                + (f"  [tailed to last {tail_n}]" if tailed else "")
                + " ---"
            )
            body = self._format_records(records)
            text = f"{header}\n{body}"

        self._write_to_widget(text)

    def dump_new_logs(
        self,
        human_readable: bool = True,
        limit: Optional[int] = None,
        advance_cursor: bool = True,
    ) -> str:
        with self._lock:
            start = self._read_cursor
            end = len(self._logs)
            entries = list(self._logs[start:end])
            if advance_cursor:
                self._read_cursor = end

        display_entries = entries if limit is None else entries[:limit]
        if not display_entries:
            return "(no new log entries)"
        if not human_readable:
            return "\n".join(str(r) for r in display_entries)

        count = len(display_entries)
        header = f"--- {count} new entr{'y' if count == 1 else 'ies'} (total in log: {end}) ---"
        return header + "\n" + self._format_records(display_entries)

    def dump_logs(self, human_readable: bool = True, limit: Optional[int] = None) -> str:
        with self._lock:
            snap = list(self._logs)
        records = snap if limit is None else snap[:limit]
        if not human_readable:
            return "\n".join(str(r) for r in records)
        return self._format_records(records)

    def dump_tail(self, n: int = 50, human_readable: bool = True) -> str:
        with self._lock:
            snap = list(self._logs)
        subset = snap[-n:] if n <= len(snap) else snap
        if not human_readable:
            return "\n".join(str(r) for r in subset)
        return self._format_records(subset)

    def dump_range(self, start: int, end: Optional[int] = None, human_readable: bool = True) -> str:
        with self._lock:
            snap = list(self._logs)
        end = len(snap) if end is None else end
        subset = snap[start:end]
        if not human_readable:
            return "\n".join(str(r) for r in subset)
        return self._format_records(subset)

    def _row_for_csv(
        self,
        r: Dict[str, Any],
        detail_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        row = {
            "time": r.get("time", 0.0),
            "node_id": r.get("node_id", ""),
            "layer": r.get("layer", ""),
            "event": r.get("event", ""),
            "packet_seq": r.get("packet_seq", ""),
            "net_seq": r.get("net_seq", ""),
            "tp_seq": r.get("tp_seq", ""),
            "frame_seq": r.get("frame_seq", ""),
            "tx_seq": r.get("tx_seq", ""),
            "ftype": r.get("ftype", ""),
            "src": r.get("src", ""),
            "dst": r.get("dst", ""),
            "next_hop": r.get("next_hop", ""),
            "details": self._safe_json(r.get("details", {})),
        }

        details = r.get("details", {})
        if isinstance(details, dict) and detail_keys:
            for k in detail_keys:
                row[f"detail_{k}"] = details.get(k, "")

        return row

    def _csv_fieldnames(self, detail_keys: Optional[List[str]] = None) -> List[str]:
        fields = list(self._CSV_FIELDS)
        if detail_keys:
            fields.extend([f"detail_{k}" for k in detail_keys])
        return fields

    def _write_csv(
        self,
        path: str,
        records: List[Dict[str, Any]],
        detail_keys: Optional[List[str]] = None,
    ) -> int:
        fieldnames = self._csv_fieldnames(detail_keys)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in records:
                w.writerow(self._row_for_csv(r, detail_keys=detail_keys))
        return len(records)

    def export_logs_csv(
        self,
        path: str,
        *,
        flatten_details: bool = True,
        detail_keys: Optional[List[str]] = None,
    ) -> int:
        with self._lock:
            snap = list(self._logs)

        keys = detail_keys
        if flatten_details and keys is None:
            keys = list(self._DEFAULT_DETAIL_EXPORT_KEYS)

        return self._write_csv(path, snap, detail_keys=keys)

    def export_new_logs_csv(
        self,
        path: str,
        advance_cursor: bool = True,
        *,
        flatten_details: bool = True,
        detail_keys: Optional[List[str]] = None,
    ) -> int:
        with self._lock:
            start = self._read_cursor
            end = len(self._logs)
            entries = list(self._logs[start:end])
            if advance_cursor:
                self._read_cursor = end

        keys = detail_keys
        if flatten_details and keys is None:
            keys = list(self._DEFAULT_DETAIL_EXPORT_KEYS)

        return self._write_csv(path, entries, detail_keys=keys)
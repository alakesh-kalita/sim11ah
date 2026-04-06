from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from sim11ah.phy import PhyLayer
from sim11ah.net import NetworkLayer
from sim11ah.tp import TransportLayer
from sim11ah.app import ApplicationLayer
from sim11ah.mac import MacLayer


class Node:
    def __init__(
        self,
        node_id: int,
        sim: "Simulator",
        role: str = "STA",
        pos: Optional[Tuple[float, float]] = None,
    ):
        self.node_id = int(node_id)
        self.sim = sim

        role_norm = str(role).upper()

        # Current simulator convention: AP is node 0, all others are STAs.
        if self.node_id == 0:
            self.role = "AP"
        else:
            if role_norm == "AP":
                raise ValueError(
                    f"Node {self.node_id}: only node_id 0 can have role 'AP' "
                    f"in the current simulator design"
                )
            self.role = "STA"

        if pos is None:
            self.pos: Tuple[float, float] = (0.0, 0.0)
        else:
            p = tuple(map(float, pos))
            if len(p) != 2:
                raise ValueError(f"Node {self.node_id}: pos must be a 2-tuple (x, y)")
            self.pos = (p[0], p[1])

        self.cfg: Dict[str, Any] = {}

        self.phy: Optional[PhyLayer] = None
        self.mac: Optional[MacLayer] = None
        self.net: Optional[NetworkLayer] = None
        self.transport: Optional[TransportLayer] = None
        self.app: Optional[ApplicationLayer] = None

        self._built: bool = False
        self._started: bool = False
        self._stopped: bool = False
        self._finalized: bool = False

    # ------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------

    @property
    def now(self) -> float:
        return float(self.sim.engine.now)

    @property
    def rng(self):
        return self.sim.engine.rng

    @property
    def is_ap(self) -> bool:
        return self.role == "AP"

    @property
    def is_sta(self) -> bool:
        return self.role == "STA"

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def build_layers(self, cfg: Dict[str, Any]) -> None:
        if self._built:
            raise RuntimeError(f"Node {self.node_id}: build_layers() called more than once")
        if self._started:
            raise RuntimeError(f"Node {self.node_id}: cannot build after start()")

        self.cfg = copy.deepcopy(cfg)

        self.phy = PhyLayer(self, self.cfg)
        self.mac = MacLayer(self, self.cfg)
        self.net = NetworkLayer(self, self.cfg)
        self.transport = TransportLayer(self, self.cfg)
        self.app = ApplicationLayer(self, self.cfg)

        for name, layer in (
            ("phy", self.phy),
            ("mac", self.mac),
            ("net", self.net),
            ("transport", self.transport),
            ("app", self.app),
        ):
            if layer is None:
                raise RuntimeError(f"Node {self.node_id}: failed to build layer '{name}'")

        self._built = True
        self._started = False
        self._stopped = False
        self._finalized = False

        for name, layer in (
            ("phy", self.phy),
            ("mac", self.mac),
            ("net", self.net),
            ("transport", self.transport),
            ("app", self.app),
        ):
            fn = getattr(layer, "post_build", None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    raise RuntimeError(
                        f"Node {self.node_id}: post_build failed in layer '{name}'"
                    ) from e

    def start(self) -> None:
        if not self._built:
            raise RuntimeError(f"Node {self.node_id}: start() called before build_layers()")
        if self._started:
            return

        for name, layer in (
            ("phy", self.phy),
            ("mac", self.mac),
            ("net", self.net),
            ("transport", self.transport),
            ("app", self.app),
        ):
            fn = getattr(layer, "start", None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    raise RuntimeError(
                        f"Node {self.node_id}: start failed in layer '{name}'"
                    ) from e

        self._started = True
        self._stopped = False

    def stop(self) -> None:
        if not self._built or self._stopped:
            return

        for name, layer in (
            ("app", self.app),
            ("transport", self.transport),
            ("net", self.net),
            ("mac", self.mac),
            ("phy", self.phy),
        ):
            fn = getattr(layer, "stop", None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    raise RuntimeError(
                        f"Node {self.node_id}: stop failed in layer '{name}'"
                    ) from e

        self._stopped = True
        # Keep _started=True to indicate the node has been started before.

    def finalize(self) -> None:
        if not self._built or self._finalized:
            return

        for name, layer in (
            ("phy", self.phy),
            ("mac", self.mac),
            ("net", self.net),
            ("transport", self.transport),
            ("app", self.app),
        ):
            fn = getattr(layer, "finalize", None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    raise RuntimeError(
                        f"Node {self.node_id}: finalize failed in layer '{name}'"
                    ) from e

        self._finalized = True

    # ------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------

    def debug_state(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "node_id": self.node_id,
            "role": self.role,
            "pos": self.pos,
            "now": self.now,
            "built": self._built,
            "started": self._started,
            "stopped": self._stopped,
            "finalized": self._finalized,
        }

        for name, layer in (
            ("phy", self.phy),
            ("mac", self.mac),
            ("net", self.net),
            ("transport", self.transport),
            ("app", self.app),
        ):
            if layer is not None:
                fn = getattr(layer, "debug_state", None)
                if callable(fn):
                    try:
                        out[name] = fn()
                    except Exception as e:
                        out[name] = {"debug_state_error": str(e)}

        return out
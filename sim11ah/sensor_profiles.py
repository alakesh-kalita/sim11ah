"""
Named sensor-type traffic presets per deployment environment (matching
ui/topology_canvas.py's ENVIRONMENTS: Open Area, Paddy Field, Smart City,
Industrial Site, Military Zone).

Each profile is just a concrete app-layer traffic-parameter bundle
standing in for what that kind of sensor would realistically send over a
HaLow link -- picking "Soil Moisture Sensor" configures the exact same
underlying traffic models (sim11ah/app.py's PeriodicTraffic/OnOffTraffic/
BurstyTraffic/"video") a manual --traffic/--packet-size/--periodic-
interval combination would. This module adds no new mechanics, just
realistic, documented defaults so you don't have to work out plausible
values for each sensor class yourself.
"""
from typing import Any, Dict

SENSOR_PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Open Area": {
        "Generic IoT Sensor": {
            "traffic": "periodic",
            "periodic_interval": 5.0,
            "periodic_jitter_s": 1.0,
            "packet_size_bytes": 128,
            "traffic_type": "sensor",
        },
    },
    "Paddy Field": {
        # Soil moisture/temperature/pH changes slowly -- a real deployment
        # reports every several minutes, not seconds, and the payload is a
        # handful of small readings plus a battery-level byte.
        "Soil Moisture Sensor": {
            "traffic": "periodic",
            "periodic_interval": 600.0,
            "periodic_jitter_s": 30.0,
            "packet_size_bytes": 24,
            "traffic_type": "telemetry",
        },
    },
    "Smart City": {
        # Mostly idle; when air quality crosses a threshold it reports a
        # short, denser burst of readings, then falls quiet again --
        # OnOffTraffic's on/off state machine models this directly.
        "Gas / Air Quality Sensor": {
            "traffic": "onoff",
            "onoff_lambda_on": 5.0,
            "onoff_on_time_s": 10.0,
            "onoff_off_time_s": 300.0,
            "packet_size_bytes": 48,
            "traffic_type": "alarm",
        },
        "Smart Meter": {
            "traffic": "periodic",
            "periodic_interval": 60.0,
            "periodic_jitter_s": 5.0,
            "packet_size_bytes": 96,
            "traffic_type": "telemetry",
        },
    },
    "Industrial Site": {
        # See ApplicationLayer._build_traffic_model's "video" branch --
        # GOP-like P/I frame size mix at a HaLow-sustainable frame rate.
        "Security Camera": {
            "traffic": "video",
            "video_fps": 5.0,
            "traffic_type": "video",
        },
        "Vibration / Temperature Sensor": {
            "traffic": "periodic",
            "periodic_interval": 30.0,
            "periodic_jitter_s": 2.0,
            "packet_size_bytes": 64,
            "traffic_type": "telemetry",
        },
    },
    "Military Zone": {
        # Long quiet stretches, then a quick run of detection reports when
        # something actually trips the sensor -- BurstyTraffic's
        # burst-then-off-time shape, not a steady periodic report.
        "Acoustic / Motion Sensor": {
            "traffic": "bursty",
            "burst_size": 5,
            "burst_intra_gap_s": 0.05,
            "burst_off_time_s": 120.0,
            "packet_size_bytes": 96,
            "traffic_type": "alarm",
        },
    },
}


def list_environments() -> list:
    return list(SENSOR_PROFILES.keys())


def list_profiles(environment: str) -> list:
    return list(SENSOR_PROFILES.get(environment, {}).keys())


def get_profile(environment: str, name: str) -> Dict[str, Any]:
    envs = SENSOR_PROFILES.get(environment)
    if not envs or name not in envs:
        raise KeyError(
            f"No sensor profile {name!r} for environment {environment!r}. "
            f"Available: {list_profiles(environment)}"
        )
    return dict(envs[name])


def apply_profile(app_cfg: Dict[str, Any], environment: str, name: str) -> Dict[str, Any]:
    """Merge a named sensor profile's params into an app config dict
    (mutates and returns app_cfg) -- profile keys override whatever was
    already there; every other app_cfg key is left untouched."""
    app_cfg.update(get_profile(environment, name))
    return app_cfg

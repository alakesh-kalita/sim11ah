import os
from sim11ah.logger import SimLogger


def safe_export_csv(logger: SimLogger, path: str) -> str:
    path = os.path.expanduser(path)
    if path.endswith(os.sep) or (os.altsep and path.endswith(os.altsep)):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "sim_logs.csv")
    elif os.path.isdir(path):
        path = os.path.join(path, "sim_logs.csv")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    logger.export_logs_csv(path)
    return path

# Module 7 — file-based event bus: training appends JSONL metrics + latest demo frame.
import json
import os
import tempfile


class EventBus:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.demo_path = os.path.join(run_dir, "demo.json")
        self.status_path = os.path.join(run_dir, "status.json")

    def log_metrics(self, record):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def write_demo(self, demo):
        _atomic_write(self.demo_path, json.dumps(demo))

    def write_status(self, status):
        _atomic_write(self.status_path, json.dumps(status))

    def read_metrics(self):
        if not os.path.exists(self.metrics_path):
            return []
        out = []
        with open(self.metrics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return out

    def read_demo(self):
        return _read_json(self.demo_path)

    def read_status(self):
        return _read_json(self.status_path)

    def reset(self):
        for p in (self.metrics_path, self.demo_path, self.status_path):
            if os.path.exists(p):
                os.remove(p)


def _atomic_write(path, text):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

import importlib.util
from pathlib import Path
import sys

services_dir = Path(__file__).resolve().parent

mapping = {
    "anomaly_engine": "anomaly-engine",
    "feature_engine": "feature-engine",
    "incident_engine": "incident-engine",
    "forecast_engine": "forecast-engine",
    "telemetry_demo": "telemetry-demo",
}

for mod_name, dir_name in mapping.items():
    engine_dir = services_dir / dir_name
    if str(engine_dir) not in sys.path:
        sys.path.insert(0, str(engine_dir))

    full_name = f"services.{mod_name}"
    init_file = engine_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        full_name,
        str(init_file) if init_file.exists() else None,
        submodule_search_locations=[str(engine_dir)],
    )
    if spec:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        setattr(sys.modules[__name__], mod_name, mod)
        if spec.loader:
            spec.loader.exec_module(mod)

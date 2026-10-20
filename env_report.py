# env_report.py - records the software environment and model/dataset revisions of a run (needs Internet for the hub queries).
# Usage (Kaggle cell or shell):  python env_report.py   -> prints and writes env_report.json
import sys, json, platform, datetime, importlib
MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B",
          "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "EleutherAI/pythia-410m", "facebook/opt-350m"]
DATASETS = ["allenai/ai2_arc", "allenai/sciq"]
rep = {"date_utc": datetime.datetime.utcnow().isoformat() + "Z", "python": sys.version, "platform": platform.platform(), "packages": {}, "models": {}, "datasets": {}}
for pkg in ["torch", "transformers", "datasets", "huggingface_hub", "numpy", "scipy", "pandas", "sklearn", "statsmodels", "matplotlib", "tokenizers"]:
    try: m = importlib.import_module(pkg); rep["packages"][pkg] = getattr(m, "__version__", "unknown")
    except Exception as e: rep["packages"][pkg] = f"not installed ({e.__class__.__name__})"
try:
    import torch; rep["cuda"] = {"available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cuda_version": torch.version.cuda}
except Exception: pass
try:
    from huggingface_hub import HfApi; api = HfApi()
    for m in MODELS:
        try: rep["models"][m] = api.model_info(m).sha
        except Exception as e: rep["models"][m] = f"unavailable ({e.__class__.__name__})"
    for d in DATASETS:
        try: rep["datasets"][d] = api.dataset_info(d).sha
        except Exception as e: rep["datasets"][d] = f"unavailable ({e.__class__.__name__})"
except Exception as e: rep["hub_error"] = str(e)
json.dump(rep, open("env_report.json", "w"), indent=2); print(json.dumps(rep, indent=2))

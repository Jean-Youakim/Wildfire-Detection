"""Streamlit demo for the wildfire detector.

Upload one or more images and get:
    * annotated detections (fire / smoke) with confidences,
    * the operational alert state per image,
    * per-class confidence, inference timing, and a model info panel,
    * downloadable results (JSON / CSV) for batch runs.

Run locally:
    streamlit run app/streamlit_app.py

The app reads defaults from ``configs/inference.yaml`` and lets you switch the
operating mode (high_recall / balanced / high_precision), adjust per-class
thresholds, and point at different weights at runtime.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

# Make the repo importable when Streamlit runs this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.data.dfire import CLASS_NAMES  # noqa: E402
from src.inference.detector import AlertState, WildfireDetector  # noqa: E402
from src.utils import load_config  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference.yaml"

ALERT_STYLE = {
    AlertState.NONE: ("No fire or smoke detected", "#0aa10a"),
    AlertState.SMOKE: ("SMOKE detected", "#1e90ff"),
    AlertState.FIRE: ("FIRE detected", "#dc143c"),
    AlertState.FIRE_AND_SMOKE: ("FIRE and SMOKE detected", "#dc143c"),
}

MODE_DESCRIPTIONS = {
    "high_recall": "catch as much as possible (early warning); more false alarms",
    "balanced": "default trade-off",
    "high_precision": "minimize false alarms; may miss faint/small events",
}


def resolve_weights_path(weights: str) -> Path:
    """Resolve a possibly-relative weights path against the repository root."""
    p = Path(weights).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


@st.cache_resource(show_spinner="Loading model...")
def load_detector(weights: str) -> WildfireDetector:
    """Load the detector once per weights file (thresholds are set per run)."""
    return WildfireDetector.from_config(DEFAULT_CONFIG, weights=weights)


def decode_upload(uploaded) -> np.ndarray | None:
    """Decode an uploaded file into an RGB array, honoring EXIF orientation."""
    try:
        image = Image.open(io.BytesIO(uploaded.getvalue()))
        image = ImageOps.exif_transpose(image)
        return np.asarray(image.convert("RGB"))
    except Exception:  # noqa: BLE001 - any decode failure means "unusable upload"
        return None


def alert_banner(alert: AlertState) -> None:
    label, color = ALERT_STYLE[alert]
    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background:{color};color:white;"
        f"font-size:18px;font-weight:700;text-align:center;'>{label}</div>",
        unsafe_allow_html=True,
    )


def analyze_image(detector: WildfireDetector, rgb: np.ndarray, filename: str) -> dict:
    """Run detection on one RGB image and package everything the UI needs."""
    detections = detector.predict(rgb)
    alert = detector.alert_state(detections)
    annotated = detector.annotate(rgb, detections)
    max_conf = {name: 0.0 for name in CLASS_NAMES.values()}
    for d in detections:
        max_conf[d.name] = max(max_conf[d.name], d.conf)
    return {
        "file": filename,
        "alert": alert,
        "detections": detections,
        "annotated": annotated,
        "max_conf": max_conf,
        "inference_ms": detector.last_inference_ms or 0.0,
    }


def result_record(result: dict) -> dict:
    """Flatten a result into a JSON/CSV-friendly record."""
    return {
        "file": result["file"],
        "alert": result["alert"].value,
        "num_detections": len(result["detections"]),
        "max_conf_smoke": round(result["max_conf"]["smoke"], 4),
        "max_conf_fire": round(result["max_conf"]["fire"], 4),
        "inference_ms": round(result["inference_ms"], 1),
        "detections": [d.as_dict() for d in result["detections"]],
    }


def render_single_result(result: dict, rgb: np.ndarray) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(rgb, use_container_width=True)
    with col2:
        st.subheader("Detections")
        st.image(result["annotated"], use_container_width=True)

    alert_banner(result["alert"])
    st.caption(f"Inference time: {result['inference_ms']:.0f} ms")

    if result["detections"]:
        st.subheader("Detection details")
        st.dataframe(
            [
                {"class": d.name, "confidence": round(d.conf, 3), "box_xyxy": str(d.xyxy)}
                for d in result["detections"]
            ],
            use_container_width=True,
        )
    else:
        st.write("No detections above the current thresholds.")


def render_batch_results(results: list[dict]) -> None:
    st.subheader(f"Batch summary — {len(results)} images")
    records = [result_record(r) for r in results]
    summary = pd.DataFrame(
        [{k: v for k, v in rec.items() if k != "detections"} for rec in records]
    )
    st.dataframe(summary, use_container_width=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download results (JSON)",
            data=json.dumps(records, indent=2),
            file_name="wildfire_predictions.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download summary (CSV)",
            data=summary.to_csv(index=False),
            file_name="wildfire_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.subheader("Annotated images")
    n_cols = 3
    cols = st.columns(n_cols)
    for i, r in enumerate(results):
        with cols[i % n_cols]:
            st.image(r["annotated"], use_container_width=True)
            label, color = ALERT_STYLE[r["alert"]]
            st.markdown(
                f"<div style='font-size:13px;'><b>{r['file']}</b><br>"
                f"<span style='color:{color};font-weight:700;'>{label}</span>"
                f" — {len(r['detections'])} detection(s), {r['inference_ms']:.0f} ms</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")


def sidebar_settings(cfg: dict) -> tuple[str, dict[str, float]]:
    """Render sidebar controls; return (weights_path_text, thresholds)."""
    default_weights = cfg.get("weights", "outputs/weights/yolo_main/weights/best.pt")
    modes = cfg.get("modes", {})

    st.header("Settings")
    weights = st.text_input(
        "Weights path",
        value=default_weights,
        help="Relative paths are resolved against the repository root.",
    )
    mode_names = list(modes) or ["balanced"]
    default_index = mode_names.index("balanced") if "balanced" in mode_names else 0
    mode = st.selectbox("Operating mode", mode_names, index=default_index)
    if mode in MODE_DESCRIPTIONS:
        st.caption(f"**{mode}**: {MODE_DESCRIPTIONS[mode]}")

    mode_conf = modes.get(mode, cfg.get("conf", {}))
    thresholds: dict[str, float] = {}
    with st.expander("Per-class thresholds", expanded=False):
        st.caption("Detections below a class's threshold are discarded.")
        for name in CLASS_NAMES.values():
            thresholds[name] = st.slider(
                f"{name} confidence",
                min_value=0.05,
                max_value=0.95,
                value=float(mode_conf.get(name, 0.25)),
                step=0.05,
            )
    return weights, thresholds


def sidebar_model_info(detector: WildfireDetector, weights_path: Path) -> None:
    st.divider()
    st.subheader("Model")
    st.markdown(
        f"- **Weights**: `{weights_path.name}`\n"
        f"- **Device**: `{detector.device}`\n"
        f"- **Input size**: {detector.imgsz}px\n"
        f"- **NMS IoU**: {detector.iou}\n"
        f"- **Classes**: {', '.join(CLASS_NAMES.values())}\n"
        f"- **Thresholds**: smoke {detector.conf['smoke']:.2f} / fire {detector.conf['fire']:.2f}"
    )


def main() -> None:
    st.set_page_config(page_title="Wildfire Detection (D-Fire)", page_icon=":fire:", layout="wide")
    st.title("Wildfire Detection — Fire & Smoke")
    st.caption("YOLO detector trained on the D-Fire dataset. Classes: 0=smoke, 1=fire.")

    cfg = load_config(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else {}

    with st.sidebar:
        weights_text, thresholds = sidebar_settings(cfg)

    weights_path = resolve_weights_path(weights_text)
    if not weights_path.is_file():
        st.warning(
            f"Weights not found at `{weights_path}`. Train a model first "
            "(see README quickstart) or point the sidebar at a valid .pt file."
        )
        return

    try:
        detector = load_detector(str(weights_path))
    except Exception as exc:  # noqa: BLE001 - surface any load failure in the UI
        st.error(f"Failed to load detector: {exc}")
        return
    # The model is cached per weights file; thresholds are cheap to apply per
    # rerun so slider changes never trigger a model reload.
    detector.conf.update(thresholds)

    with st.sidebar:
        sidebar_model_info(detector, weights_path)

    uploads = st.file_uploader(
        "Upload image(s) — drag & drop supported",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
    if not uploads:
        st.info("Upload one or more images to run detection.")
        return

    results: list[dict] = []
    skipped: list[str] = []
    progress = st.progress(0.0, text="Running detection...") if len(uploads) > 1 else None
    last_rgb: np.ndarray | None = None

    for i, uploaded in enumerate(uploads):
        rgb = decode_upload(uploaded)
        if rgb is None:
            skipped.append(uploaded.name)
            continue
        with st.spinner(f"Analyzing {uploaded.name}..."):
            results.append(analyze_image(detector, rgb, uploaded.name))
        last_rgb = rgb
        if progress is not None:
            progress.progress(
                (i + 1) / len(uploads), text=f"Analyzed {i + 1}/{len(uploads)} images"
            )

    if progress is not None:
        progress.empty()
    if skipped:
        st.error(f"Could not decode: {', '.join(skipped)}")
    if not results:
        return

    if len(results) == 1:
        render_single_result(results[0], last_rgb)
    else:
        render_batch_results(results)


if __name__ == "__main__":
    main()

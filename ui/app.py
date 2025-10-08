# ui/app.py
# --- resilient imports: work both as package and as script ---
import streamlit as st

try:
    # package mode (preferred)
    from .paths import get_root, get_models_dir
    from .config import APP_TITLE, APP_CAPTION, CHART_HEIGHT
    from . import config
    from .inference import (
        discover_models,
        resolve_meta,
        load_model,
        load_meta,
        predict_proba,
    )
    from .components import (
        render_header,
        render_model_picker,
        render_text_input,
        flash_top_prediction,
        render_footer,
    )
    from .charts import probability_bar_chart
except ImportError:  # script mode fallback
    from paths import get_root, get_models_dir
    from config import APP_TITLE, APP_CAPTION, CHART_HEIGHT
    import config
    from inference import (
        discover_models,
        resolve_meta,
        load_model,
        load_meta,
        predict_proba,
    )
    from components import (
        render_header,
        render_model_picker,
        render_text_input,
        flash_top_prediction,
        render_footer,
    )
    from charts import probability_bar_chart



def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="centered")

    # Header
    render_header(APP_TITLE, APP_CAPTION)

    # Paths
    root: Path = get_root(__file__)
    models_dir: Path = get_models_dir(root)

    # Discover available models
    model_paths = discover_models(models_dir)
    if not model_paths:
        st.error(f"No .joblib models found in {models_dir}")
        st.stop()

    # Model picker
    model_path = render_model_picker(model_paths)

    # Resolve metadata
    try:
        meta_path, tried = resolve_meta(model_path, models_dir)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Load model + metadata
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    try:
        meta = load_meta(meta_path)
        labels = list(meta["label_cols"])
        threshold = float(meta.get("threshold", config.DEFAULT_THRESHOLD))
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Text input + Predict
    text, do_predict = render_text_input()
    if do_predict:
        if not text:
            st.warning("Please enter some text.")
        else:
            probs = predict_proba(model, text)  # shape: (n_labels,)
            # Flash summary (flagged / no toxicity)
            _, order = flash_top_prediction(probs, labels, threshold)

            # Probability chart
            chart = probability_bar_chart(labels, probs, height=CHART_HEIGHT)
            st.altair_chart(chart, use_container_width=True)

    # Footer
    render_footer(model_path.name, meta_path.name, threshold)


if __name__ == "__main__":
    main()

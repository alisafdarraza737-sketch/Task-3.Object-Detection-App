from functools import lru_cache
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="wide",
)

MODEL_NAME = "yolov8n.pt"


@st.cache_resource
def load_model() -> YOLO:
    """Load the lightweight YOLOv8 model once per Streamlit session."""
    return YOLO(MODEL_NAME)


def read_image(file_bytes: bytes) -> Image.Image:
    """Decode an uploaded image and normalize it to RGB."""
    try:
        return Image.open(BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a supported image.") from exc


def main() -> None:
    st.title("🎯 Object Detection with YOLO")
    st.write(
        "Detect and label objects in an uploaded image or a picture captured "
        "with your device camera."
    )

    with st.sidebar:
        st.header("Detection settings")
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
        )
        st.caption(f"Model: {MODEL_NAME} (downloads automatically on first run)")

    upload_tab, camera_tab = st.tabs(["Upload image", "Use camera"])

    with upload_tab:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
        )
        image = None
        if uploaded is not None:
            try:
                image = read_image(uploaded.getvalue())
            except ValueError as exc:
                st.error(str(exc))

    with camera_tab:
        camera_file = st.camera_input("Take a picture")
        camera_image = None
        if camera_file is not None:
            try:
                camera_image = read_image(camera_file.getvalue())
            except ValueError as exc:
                st.error(str(exc))

    selected_image = image if image is not None else camera_image
    if selected_image is None:
        st.info("Upload an image or take a camera picture to begin.")
        return

    try:
        model = load_model()
        results = model.predict(
            source=selected_image,
            conf=confidence,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()
        annotated_rgb = Image.fromarray(annotated[:, :, ::-1])

        left, right = st.columns(2)
        with left:
            st.subheader("Original image")
            st.image(selected_image, use_container_width=True)
        with right:
            st.subheader("Detection result")
            st.image(annotated_rgb, use_container_width=True)

        boxes = result.boxes
        count = len(boxes) if boxes is not None else 0
        st.metric("Objects detected", count)

        if count:
            names = result.names
            rows = []
            for box in boxes:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                rows.append(
                    {
                        "Object": names[class_id],
                        "Confidence": f"{score:.1%}",
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No objects were detected above the selected threshold.")

    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        st.caption("Check your internet connection for the first model download and try again.")


if __name__ == "__main__":
    main()

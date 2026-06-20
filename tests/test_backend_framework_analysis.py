from pathlib import Path


DOC = Path(__file__).resolve().parents[1] / "docs" / "BACKEND_FRAMEWORK_ANALYSIS.md"


def test_backend_framework_analysis_exists_and_recommends_no_rewrite():
    text = DOC.read_text(encoding="utf-8")

    assert "Recommendation: keep Python for ML-serving" in text
    assert "Do not rewrite the ML-serving backend in Rust, C#, or Go" in text
    assert "GPU model inference" in text


def test_backend_framework_analysis_covers_current_stack_and_migration_options():
    text = DOC.read_text(encoding="utf-8")

    for required in [
        "src/web/backend/app.py",
        "src/web/backend/video_filter_api.py",
        "src/web/mediasoup",
        "Rust",
        "C#",
        "Go",
        "FastAPI",
        "gunicorn",
        "ONNX",
    ]:
        assert required in text

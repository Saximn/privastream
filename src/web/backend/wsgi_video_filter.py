"""WSGI entrypoint for the video filter API."""

from src.web.backend.video_filter_api import app

application = app


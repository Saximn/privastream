"""WSGI entrypoint for the room orchestration service."""

from src.web.backend.app import create_app

app, socketio = create_app()

# Some WSGI hosts look for `application`.
application = app


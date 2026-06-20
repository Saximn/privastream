"""Web interface components for Privastream."""

__all__ = ["create_app", "RoomManager"]


def __getattr__(name):
    if name in __all__:
        from .backend.app import RoomManager, create_app

        return {"create_app": create_app, "RoomManager": RoomManager}[name]
    raise AttributeError(name)

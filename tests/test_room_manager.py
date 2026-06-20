from src.web.backend.app import RoomManager


def test_room_manager_create_join_leave_and_duplicate_viewer():
    manager = RoomManager(room_ttl_seconds=60)
    manager.add_user("host-sid", "host-user")
    manager.add_user("viewer-sid", "viewer-user")

    room_id = manager.create_room("host-sid")

    assert manager.join_room(room_id, "viewer-sid") is True
    assert manager.join_room(room_id, "viewer-sid") is True
    assert manager.get_room_info(room_id)["viewers"] == ["viewer-sid"]

    manager.leave_room(room_id, "viewer-sid")
    assert manager.get_room_info(room_id)["viewers"] == []


def test_room_manager_host_leave_deletes_room_and_user_room_state():
    manager = RoomManager(room_ttl_seconds=60)
    manager.add_user("host-sid", "host-user", role="host", room=None)
    room_id = manager.create_room("host-sid")
    manager.users["host-sid"]["room"] = room_id

    manager.leave_room(room_id, "host-sid", is_host=True)

    assert manager.get_room_info(room_id) is None
    assert manager.users["host-sid"]["room"] is None
    assert manager.users["host-sid"]["role"] is None


def test_room_manager_cleanup_expired_rooms():
    manager = RoomManager(room_ttl_seconds=1)
    room_id = manager.create_room("host-sid")
    manager.rooms[room_id]["expires_at"] = 0

    assert manager.cleanup_expired_rooms() == 1
    assert room_id not in manager.rooms


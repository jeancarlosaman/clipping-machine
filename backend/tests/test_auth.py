def test_missing_auth_header_rejected(client):
    resp = client.get("/api/v1/stream-jobs")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthenticated"


def test_malformed_auth_header_rejected(client):
    resp = client.get("/api/v1/stream-jobs", headers={"Authorization": "Basic abc123"})
    assert resp.status_code == 401


def test_garbage_token_rejected(client):
    resp = client.get("/api/v1/stream-jobs", headers={"Authorization": "Bearer not-a-real-jwt"})
    assert resp.status_code == 401


def test_valid_token_accepted(client, auth_headers):
    resp = client.get("/api/v1/stream-jobs", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_oauth_state_token_round_trips():
    import uuid

    from app.core.auth import create_oauth_state_token, decode_oauth_state_token

    user_id = str(uuid.uuid4())
    token = create_oauth_state_token(user_id)
    assert decode_oauth_state_token(token) == user_id


def test_oauth_state_token_rejects_a_real_access_token():
    # Both are HS256-signed with the same jwt_secret -- decode_oauth_state_token
    # must still refuse a real access token passed where a state token is
    # expected (the `typ` claim is the only thing distinguishing them).
    import pytest

    from app.core.auth import create_access_token, decode_oauth_state_token

    access_token = create_access_token("some-user-id")
    with pytest.raises(ValueError):
        decode_oauth_state_token(access_token)


def test_access_token_rejects_an_oauth_state_token():
    import pytest

    from app.core.auth import create_oauth_state_token, decode_access_token

    state_token = create_oauth_state_token("some-user-id")
    # Both are HS256-signed with the same jwt_secret -- only the `typ`
    # claim tells them apart, so decode_access_token must reject a state
    # token even though its signature is perfectly valid.
    with pytest.raises(ValueError):
        decode_access_token(state_token)


def test_bearer_auth_rejects_an_oauth_state_token_used_as_access_token(client, user):
    # End-to-end version of the unit test above, through the real
    # dependency chain (app.api.deps.get_current_user) rather than calling
    # decode_access_token directly.
    from app.core.auth import create_oauth_state_token

    state_token = create_oauth_state_token(str(user.id))
    resp = client.get("/api/v1/stream-jobs", headers={"Authorization": f"Bearer {state_token}"})
    assert resp.status_code == 401

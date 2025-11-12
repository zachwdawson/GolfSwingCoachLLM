import io
import uuid
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.api.upload import get_db
from app.main import app
from app.models.video import Base, Video

TEST_DB_URL = "sqlite:///./test.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(test_db):
    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def mock_s3_client():
    with patch("app.api.upload.s3_client") as mock:
        mock.upload_file.return_value = True
        yield mock


def test_upload_video_success(client, mock_s3_client, test_db):
    """Test successful video upload."""
    video_content = b"fake video content" * 1000  # ~17KB
    video_file = io.BytesIO(video_content)
    video_file.name = "test_video.mp4"

    response = client.post(
        "/upload",
        files={"file": ("test_video.mp4", video_file, "video/mp4")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "video_id" in data
    assert "s3_key" in data
    assert data["s3_key"].startswith("videos/")
    # Video is no longer uploaded to S3 immediately - it's saved to temp file
    # and will be uploaded after processing completes
    mock_s3_client.upload_file.assert_not_called()
    
    # Verify video record was created in database
    video_id = uuid.UUID(data["video_id"])
    db = TestSessionLocal()
    video = db.query(Video).filter(Video.id == video_id).first()
    assert video is not None
    assert video.s3_key == data["s3_key"]
    assert video.status == "uploaded"
    db.close()


def test_upload_video_invalid_mime_type(client):
    """Test upload with invalid MIME type returns 415."""
    video_content = b"fake video content"
    video_file = io.BytesIO(video_content)
    video_file.name = "test_video.avi"

    response = client.post(
        "/upload",
        files={"file": ("test_video.avi", video_file, "video/x-msvideo")},
    )

    assert response.status_code == 415
    assert "Unsupported media type" in response.json()["detail"]


def test_upload_video_too_large(client):
    """Test upload with file too large returns 413."""
    video_content = b"x" * (101 * 1024 * 1024)  # 101MB
    video_file = io.BytesIO(video_content)
    video_file.name = "test_video.mp4"

    response = client.post(
        "/upload",
        files={"file": ("test_video.mp4", video_file, "video/mp4")},
    )

    assert response.status_code == 413  # HTTP_413_CONTENT_TOO_LARGE
    assert "exceeds" in response.json()["detail"].lower()


def test_get_video_not_found(client):
    """Test getting non-existent video returns 404."""
    video_id = uuid.uuid4()
    response = client.get(f"/videos/{video_id}")
    assert response.status_code == 404


def test_get_video_success(client, mock_s3_client, test_db):
    """Test getting existing video."""
    video_id = uuid.uuid4()
    s3_key = f"videos/{video_id}/raw.mp4"
    video = Video(id=video_id, s3_key=s3_key, status="uploaded")

    db = TestSessionLocal()
    db.add(video)
    db.commit()
    db.close()

    response = client.get(f"/videos/{video_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == str(video_id)
    assert data["status"] == "uploaded"
    assert data["s3_key"] == s3_key


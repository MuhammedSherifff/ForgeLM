import gzip
import io

import numpy as np

from data_pipeline.pretrain_data import check_shards, fetch_python_blob


def write_shard(path, values):
    np.asarray(values, dtype=np.uint16).tofile(path)


def test_valid_shards_are_reported_as_passed(tmp_path):
    source_dir = tmp_path / "fineweb-edu-dedup"
    source_dir.mkdir()
    write_shard(source_dir / "fineweb-edu-dedup_0000.bin", [10, 2, 11, 2])

    report = check_shards(tmp_path, vocab_size=32_000, eos_id=2, chunk_tokens=2)

    assert report["errors"] == []
    assert report["total_tokens"] == 4
    assert report["total_eos_tokens"] == 2


def test_out_of_range_token_is_an_error(tmp_path):
    source_dir = tmp_path / "python-edu"
    source_dir.mkdir()
    write_shard(source_dir / "python-edu_0000.bin", [1, 2, 32_000])

    report = check_shards(tmp_path, vocab_size=32_000, eos_id=2, chunk_tokens=2)

    assert any("vocabulary size" in error for error in report["errors"])


def test_missing_shard_index_is_an_error(tmp_path):
    source_dir = tmp_path / "cosmopedia-v2"
    source_dir.mkdir()
    write_shard(source_dir / "cosmopedia-v2_0000.bin", [1, 2])
    write_shard(source_dir / "cosmopedia-v2_0002.bin", [3, 2])

    report = check_shards(tmp_path, vocab_size=32_000, eos_id=2, chunk_tokens=2)

    assert any("missing shard indices" in error for error in report["errors"])


class _FakeBody(io.BytesIO):
    pass


class _FakeS3:
    def __init__(self, payload: bytes | None):
        self.payload = payload

    def get_object(self, Bucket, Key):
        if self.payload is None:
            raise RuntimeError("NoSuchKey")
        assert Bucket == "softwareheritage"
        assert Key.startswith("content/")
        return {"Body": _FakeBody(self.payload)}


def _gzip_bytes(text: str) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as compressed:
        compressed.write(text.encode("utf-8"))
    return buffer.getvalue()


def test_fetch_python_blob_returns_text():
    client = _FakeS3(_gzip_bytes("print('hi')"))
    assert fetch_python_blob("abc123", client=client) == "print('hi')"


def test_fetch_python_blob_failures_are_skipped():
    assert fetch_python_blob("missing", client=_FakeS3(None)) == ""
    assert fetch_python_blob("bad", client=_FakeS3(b"not-gzip")) == ""

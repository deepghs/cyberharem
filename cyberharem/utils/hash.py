import hashlib


def file_sha256(file_path, chunk_size: int = 65536):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(chunk_size)  # 64 KB
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

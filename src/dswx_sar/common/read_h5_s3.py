# s3_h5_min.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Iterator
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotoConfig
import h5py


def is_s3_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")

def is_http_url(path: str) -> bool:
    return isinstance(path, str) and (path.startswith("http://") or path.startswith("https://"))

def parse_s3_url(url: str) -> Tuple[str, str]:
    p = urlparse(url)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise ValueError(f"Invalid S3 URL: {url}")
    return p.netloc, p.path.lstrip("/")


def _discover_region(bucket: str, session: boto3.Session) -> str:
    s3 = session.client("s3", config=BotoConfig(signature_version="s3v4"))
    try:
        resp = s3.get_bucket_location(Bucket=bucket)
        loc = resp.get("LocationConstraint")
        return "us-east-1" if (loc is None) else loc
    except Exception:
        return session.region_name or "us-east-1"

def s3_exists(s3_url: str, *, profile: Optional[str] = None) -> bool:
    bucket, key = parse_s3_url(s3_url)
    sess = boto3.Session(profile_name=profile)
    s3 = sess.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def file_exists(path: str, *, profile: Optional[str] = None) -> bool:
    if is_s3_path(path):
        return s3_exists(path, profile=profile)
    if is_http_url(path):
        # For presigned URLs we can’t HEAD reliably without requests; assume reachable
        return True
    return os.path.exists(path)


class H5Reader:
    """
    Context manager that opens an HDF5 file from:
      • local path    → h5py.File(path, "r")
      • s3://…        → h5py.File(path, "r", driver="ros3", **aws_kwargs)
      • https://…     → h5py.File(url,  "r", driver="ros3")  (presigned URL)
    """

    def __init__(
        self,
        path: str,
        *,
        profile: Optional[str] = "saml-pub",
        region: Optional[str] = None,
        use_presign_fallback: bool = False,
        request_payer_requester: bool = False,
        expires_in: int = 3600,
        # HDF5 cache tuning
        rdcc_nbytes: Optional[int] = None,
        rdcc_nslots: Optional[int] = None,
        rdcc_w0: Optional[float] = None,
        # ROS3 page buffer (if supported by your HDF5 build)
        page_buf_size: Optional[int] = None,
    ):
        self.path = path
        self.profile = profile
        self.region = region
        self.use_presign_fallback = use_presign_fallback
        self.request_payer_requester = request_payer_requester
        self.expires_in = expires_in

        # Defaults
        self.rdcc_nbytes = rdcc_nbytes if rdcc_nbytes is not None else (512 << 20)  # 512 MB
        self.rdcc_nslots = rdcc_nslots if rdcc_nslots is not None else 1_000_003
        self.rdcc_w0 = rdcc_w0 if rdcc_w0 is not None else 0.75

        self.page_buf_size = page_buf_size
        self._f: Optional[h5py.File] = None

    def _open_local(self) -> h5py.File:
        return h5py.File(
            self.path, "r",
            rdcc_nbytes=self.rdcc_nbytes,
            rdcc_nslots=self.rdcc_nslots,
            rdcc_w0=self.rdcc_w0,
        )

    def _open_http(self) -> h5py.File:
        kwargs = {
            "driver": "ros3",
            "rdcc_nbytes": self.rdcc_nbytes,
            "rdcc_nslots": self.rdcc_nslots,
            "rdcc_w0": self.rdcc_w0,
        }
        if self.page_buf_size:
            kwargs["page_buf_size"] = self.page_buf_size

        return h5py.File(self.path, "r", **kwargs)

    def _open_s3_ros3(self) -> h5py.File:
        assert is_s3_path(self.path)
        bucket, key = parse_s3_url(self.path)
        sess = boto3.Session(profile_name=self.profile, region_name=self.region)
        effective_region = self.region or _discover_region(bucket, sess)

        creds = sess.get_credentials()
        if creds is None:
            raise RuntimeError("No AWS credentials available (profile/env/IMDS).")
        fc = creds.get_frozen_credentials()

        base_kwargs = {
            "driver": "ros3",
            "rdcc_nbytes": self.rdcc_nbytes,
            "rdcc_nslots": self.rdcc_nslots,
            "rdcc_w0": self.rdcc_w0,
        }
        if self.page_buf_size:
            base_kwargs["page_buf_size"] = self.page_buf_size

        # bytes variant
        kwargs_bytes = dict(base_kwargs)
        kwargs_bytes.update({
            "aws_region": (effective_region or "us-east-1").encode("utf-8"),
            "secret_id": fc.access_key.encode("utf-8"),
            "secret_key": fc.secret_key.encode("utf-8"),
        })
        if fc.token:
            kwargs_bytes["session_token"] = fc.token.encode("utf-8")

        # str variant
        kwargs_str = dict(base_kwargs)
        kwargs_str.update({
            "aws_region": (effective_region or "us-east-1"),
            "secret_id": fc.access_key,
            "secret_key": fc.secret_key,
        })
        if fc.token:
            kwargs_str["session_token"] = fc.token

        last_err: Optional[Exception] = None
        try:
            return h5py.File(self.path, "r", **kwargs_bytes)
        except Exception as e:
            last_err = e

        try:
            return h5py.File(self.path, "r", **kwargs_str)
        except Exception as e:
            last_err = e

        if self.use_presign_fallback:
            s3 = sess.client("s3", region_name=effective_region,
                             config=BotoConfig(signature_version="s3v4"))
            params = {"Bucket": bucket, "Key": key}
            if self.request_payer_requester:
                params["RequestPayer"] = "requester"
            url = s3.generate_presigned_url("get_object", Params=params,
                                            ExpiresIn=self.expires_in)
            return self._open_http_with_url(url)

        raise RuntimeError(
            f"ROS3 open failed for s3://{bucket}/{key} (region={effective_region}). "
            f"Last error: {type(last_err).__name__}: {last_err}"
        )

    def _open_http_with_url(self, url: str) -> h5py.File:
        kwargs = {
            "driver": "ros3",
            "rdcc_nbytes": self.rdcc_nbytes,
            "rdcc_nslots": self.rdcc_nslots,
            "rdcc_w0": self.rdcc_w0,
        }
        if self.page_buf_size:
            kwargs["page_buf_size"] = self.page_buf_size
        return h5py.File(url, "r", **kwargs)

    def __enter__(self) -> h5py.File:
        if is_s3_path(self.path):
            self._f = self._open_s3_ros3()
        elif is_http_url(self.path):
            self._f = self._open_http()
        else:
            self._f = self._open_local()
        return self._f

    def __exit__(self, exc_type, exc, tb):
        if self._f is not None:
            self._f.close()


def slice_gen(total_size: int, batch_size: int, combine_rem: bool = True) -> Iterator[slice]:
    n_full = total_size // batch_size
    n_total_full = n_full * batch_size
    n_rem = total_size - n_total_full
    if combine_rem and n_rem > 0 and n_total_full >= batch_size:
        for start in range(0, n_total_full - batch_size, batch_size):
            yield slice(start, start + batch_size)
        yield slice(n_total_full - batch_size, total_size)
    else:
        for start in range(0, n_total_full, batch_size):
            yield slice(start, start + batch_size)
        if n_rem:
            yield slice(n_total_full, total_size)

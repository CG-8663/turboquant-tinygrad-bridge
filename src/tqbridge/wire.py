"""Wire protocol: header serialisation, CRC32 validation, format negotiation."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

MAGIC = 0x54514B56  # "TQKV"
HEADER_VERSION = 1
HEADER_SIZE = 40  # bytes
HEADER_STRUCT = "<IBBB xHHIHHHH QI xxxx"
# Offsets:
#  0x00  u32  magic
#  0x04  u8   version
#  0x05  u8   fmt_k
#  0x06  u8   fmt_v
#  0x07  u8   reserved (pad)
#  0x08  u16  n_layers
#  0x0A  u16  layer_start
#  0x0C  u32  seq_len
#  0x10  u16  n_heads_k
#  0x12  u16  n_heads_v
#  0x14  u16  head_dim
#  0x16  u16  flags
#  0x18  u64  payload_bytes
#  0x20  u32  crc32
#  0x24  u32  reserved


class Format(IntEnum):
    """Compression format identifiers for K/V wire encoding."""
    FP16 = 0x10
    Q8_0 = 0x08
    Q5_K_M = 0x05
    TURBO4 = 0x04
    TURBO3 = 0x03
    TURBO2 = 0x02


# Flags
FLAG_ASYMMETRIC_KV = 1 << 0


@dataclass(frozen=True)
class WireHeader:
    """40-byte wire header for cross-backend KV transfers."""
    fmt_k: Format
    fmt_v: Format
    n_layers: int
    layer_start: int
    seq_len: int
    n_heads_k: int
    n_heads_v: int
    head_dim: int
    payload_bytes: int
    flags: int = 0
    version: int = HEADER_VERSION

    @property
    def asymmetric(self) -> bool:
        return bool(self.flags & FLAG_ASYMMETRIC_KV)


class WireError(Exception):
    """Raised for wire protocol violations."""


def encode_header(header: WireHeader) -> bytes:
    """Serialise a WireHeader to 40 bytes, little-endian, with CRC32."""
    flags = header.flags
    if header.fmt_k != header.fmt_v:
        flags |= FLAG_ASYMMETRIC_KV

    # Pack everything except crc32 (set to 0 for checksum computation)
    data = struct.pack(
        HEADER_STRUCT,
        MAGIC,
        header.version,
        int(header.fmt_k),
        int(header.fmt_v),
        header.n_layers,
        header.layer_start,
        header.seq_len,
        header.n_heads_k,
        header.n_heads_v,
        header.head_dim,
        flags,
        header.payload_bytes,
        0,  # crc32 placeholder
    )
    # Compute CRC32 over bytes 0x00-0x1F (first 32 bytes)
    crc = zlib.crc32(data[:0x20]) & 0xFFFFFFFF
    # Write CRC32 at offset 0x20
    data = data[:0x20] + struct.pack("<I", crc) + data[0x24:]
    assert len(data) == HEADER_SIZE
    return data


def decode_header(data: bytes) -> WireHeader:
    """Deserialise 40 bytes into a WireHeader, validating magic and CRC32."""
    if len(data) < HEADER_SIZE:
        raise WireError(f"Header too short: {len(data)} bytes, need {HEADER_SIZE}")

    data = data[:HEADER_SIZE]

    # Verify magic
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != MAGIC:
        raise WireError(f"Bad magic: 0x{magic:08X}, expected 0x{MAGIC:08X}")

    # Verify CRC32: compute over bytes 0x00-0x1F with crc32 field zeroed
    crc_stored = struct.unpack_from("<I", data, 0x20)[0]
    check_data = data[:0x20] + b"\x00\x00\x00\x00" + data[0x24:]
    crc_computed = zlib.crc32(check_data[:0x20]) & 0xFFFFFFFF
    if crc_stored != crc_computed:
        raise WireError(
            f"CRC32 mismatch: stored=0x{crc_stored:08X}, "
            f"computed=0x{crc_computed:08X}"
        )

    # Unpack all fields
    (
        _magic, version, fmt_k, fmt_v,
        n_layers, layer_start, seq_len,
        n_heads_k, n_heads_v, head_dim, flags,
        payload_bytes, _crc,
    ) = struct.unpack(HEADER_STRUCT, data)

    # Validate version
    if version != HEADER_VERSION:
        raise WireError(f"Unknown wire version: {version}, supported: {HEADER_VERSION}")

    # Validate format values
    try:
        fmt_k = Format(fmt_k)
    except ValueError:
        raise WireError(f"Unknown K format: 0x{fmt_k:02X}")
    try:
        fmt_v = Format(fmt_v)
    except ValueError:
        raise WireError(f"Unknown V format: 0x{fmt_v:02X}")

    return WireHeader(
        version=version,
        fmt_k=fmt_k,
        fmt_v=fmt_v,
        n_layers=n_layers,
        layer_start=layer_start,
        seq_len=seq_len,
        n_heads_k=n_heads_k,
        n_heads_v=n_heads_v,
        head_dim=head_dim,
        flags=flags,
        payload_bytes=payload_bytes,
    )


@dataclass(frozen=True)
class Capabilities:
    """Supported formats and limits for one endpoint."""
    formats: frozenset[Format]
    max_head_dim: int
    max_seq_len: int


@dataclass(frozen=True)
class WireConfig:
    """Negotiated wire configuration between two endpoints."""
    fmt_k: Format
    fmt_v: Format
    max_head_dim: int
    max_seq_len: int


# Format preference order for K (higher precision preferred)
_K_PREFERENCE = [Format.Q8_0, Format.Q5_K_M, Format.FP16, Format.TURBO4, Format.TURBO3]
# Format preference order for V (higher compression preferred)
_V_PREFERENCE = [Format.TURBO3, Format.TURBO4, Format.Q8_0, Format.Q5_K_M, Format.FP16]


class NegotiationError(WireError):
    """Raised when format negotiation fails."""


def negotiate_format(
    local: Capabilities,
    remote: Capabilities,
    override_k: Format | None = None,
    override_v: Format | None = None,
) -> WireConfig:
    """Select best common wire config from two endpoints' capabilities.

    Default strategy: q8_0 K + turbo3 V (asymmetric, per TheTom recommendation).
    Falls back through preference lists if defaults aren't mutually supported.
    """
    common = local.formats & remote.formats
    if not common:
        raise NegotiationError(
            f"No common formats: local={local.formats}, remote={remote.formats}"
        )

    if override_k is not None:
        if override_k not in common:
            raise NegotiationError(f"Override fmt_k={override_k.name} not in common formats")
        fmt_k = override_k
    else:
        fmt_k = _pick_best(common, _K_PREFERENCE)

    if override_v is not None:
        if override_v not in common:
            raise NegotiationError(f"Override fmt_v={override_v.name} not in common formats")
        fmt_v = override_v
    else:
        fmt_v = _pick_best(common, _V_PREFERENCE)

    return WireConfig(
        fmt_k=fmt_k,
        fmt_v=fmt_v,
        max_head_dim=min(local.max_head_dim, remote.max_head_dim),
        max_seq_len=min(local.max_seq_len, remote.max_seq_len),
    )


def _pick_best(available: frozenset[Format], preference: list[Format]) -> Format:
    for fmt in preference:
        if fmt in available:
            return fmt
    # Fallback: pick any
    return next(iter(available))

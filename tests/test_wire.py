"""Tests for wire protocol: header serialisation, CRC32, format negotiation."""

import struct
import pytest
from tqbridge.wire import (
    Format,
    WireHeader,
    WireConfig,
    Capabilities,
    encode_header,
    decode_header,
    negotiate_format,
    WireError,
    NegotiationError,
    MAGIC,
    HEADER_SIZE,
    HEADER_VERSION,
    FLAG_ASYMMETRIC_KV,
)


# --- Header round-trip ---


def _make_header(**overrides) -> WireHeader:
    defaults = dict(
        fmt_k=Format.Q8_0,
        fmt_v=Format.TURBO3,
        n_layers=40,
        layer_start=0,
        seq_len=32768,
        n_heads_k=8,
        n_heads_v=8,
        head_dim=128,
        payload_bytes=525_000_000,
    )
    defaults.update(overrides)
    return WireHeader(**defaults)


def test_encode_header_size():
    data = encode_header(_make_header())
    assert len(data) == HEADER_SIZE


def test_header_round_trip():
    h = _make_header()
    data = encode_header(h)
    h2 = decode_header(data)
    assert h2.fmt_k == h.fmt_k
    assert h2.fmt_v == h.fmt_v
    assert h2.n_layers == h.n_layers
    assert h2.layer_start == h.layer_start
    assert h2.seq_len == h.seq_len
    assert h2.n_heads_k == h.n_heads_k
    assert h2.n_heads_v == h.n_heads_v
    assert h2.head_dim == h.head_dim
    assert h2.payload_bytes == h.payload_bytes
    assert h2.version == HEADER_VERSION


def test_header_round_trip_symmetric():
    h = _make_header(fmt_k=Format.TURBO3, fmt_v=Format.TURBO3)
    data = encode_header(h)
    h2 = decode_header(data)
    assert h2.fmt_k == Format.TURBO3
    assert h2.fmt_v == Format.TURBO3
    assert not h2.asymmetric


def test_header_asymmetric_flag_set_automatically():
    h = _make_header(fmt_k=Format.Q8_0, fmt_v=Format.TURBO3)
    data = encode_header(h)
    h2 = decode_header(data)
    assert h2.asymmetric


def test_header_all_formats():
    for fmt in Format:
        h = _make_header(fmt_k=fmt, fmt_v=fmt)
        data = encode_header(h)
        h2 = decode_header(data)
        assert h2.fmt_k == fmt
        assert h2.fmt_v == fmt


def test_header_magic():
    data = encode_header(_make_header())
    magic = struct.unpack_from("<I", data, 0)[0]
    assert magic == MAGIC


def test_header_various_head_dims():
    for hd in [64, 128, 256, 512]:
        h = _make_header(head_dim=hd)
        h2 = decode_header(encode_header(h))
        assert h2.head_dim == hd


# --- CRC32 validation ---


def test_corrupted_header_rejected():
    data = bytearray(encode_header(_make_header()))
    # Flip a bit in the payload_bytes field
    data[0x18] ^= 0xFF
    with pytest.raises(WireError, match="CRC32 mismatch"):
        decode_header(bytes(data))


def test_corrupted_crc_field_rejected():
    data = bytearray(encode_header(_make_header()))
    # Corrupt the CRC directly
    data[0x20] ^= 0x01
    with pytest.raises(WireError, match="CRC32 mismatch"):
        decode_header(bytes(data))


# --- Error handling ---


def test_bad_magic_rejected():
    data = bytearray(encode_header(_make_header()))
    struct.pack_into("<I", data, 0, 0xDEADBEEF)
    # Recompute CRC so it's not a CRC error
    import zlib
    crc = zlib.crc32(bytes(data[:0x20])) & 0xFFFFFFFF
    struct.pack_into("<I", data, 0x20, crc)
    with pytest.raises(WireError, match="Bad magic"):
        decode_header(bytes(data))


def test_unknown_version_rejected():
    h = _make_header()
    data = bytearray(encode_header(h))
    data[0x04] = 99  # unknown version
    # Recompute CRC
    import zlib
    crc = zlib.crc32(bytes(data[:0x20])) & 0xFFFFFFFF
    struct.pack_into("<I", data, 0x20, crc)
    with pytest.raises(WireError, match="Unknown wire version"):
        decode_header(bytes(data))


def test_unknown_format_rejected():
    data = bytearray(encode_header(_make_header()))
    data[0x05] = 0xFE  # invalid format
    import zlib
    crc = zlib.crc32(bytes(data[:0x20])) & 0xFFFFFFFF
    struct.pack_into("<I", data, 0x20, crc)
    with pytest.raises(WireError, match="Unknown K format"):
        decode_header(bytes(data))


def test_short_data_rejected():
    with pytest.raises(WireError, match="Header too short"):
        decode_header(b"\x00" * 10)


# --- Format negotiation ---


def _caps(*fmts, max_hd=512, max_seq=131072) -> Capabilities:
    return Capabilities(
        formats=frozenset(fmts),
        max_head_dim=max_hd,
        max_seq_len=max_seq,
    )


def test_negotiate_default_asymmetric():
    """Default negotiation should prefer q8_0 K + turbo3 V."""
    local = _caps(Format.Q8_0, Format.TURBO3, Format.TURBO4, Format.FP16)
    remote = _caps(Format.Q8_0, Format.TURBO3, Format.TURBO4, Format.FP16)
    config = negotiate_format(local, remote)
    assert config.fmt_k == Format.Q8_0
    assert config.fmt_v == Format.TURBO3


def test_negotiate_falls_back():
    """If turbo3 not available, should pick next best V format."""
    local = _caps(Format.Q8_0, Format.TURBO4, Format.FP16)
    remote = _caps(Format.Q8_0, Format.TURBO4, Format.FP16)
    config = negotiate_format(local, remote)
    assert config.fmt_k == Format.Q8_0
    assert config.fmt_v == Format.TURBO4


def test_negotiate_no_common():
    local = _caps(Format.TURBO3)
    remote = _caps(Format.FP16)
    with pytest.raises(NegotiationError, match="No common formats"):
        negotiate_format(local, remote)


def test_negotiate_override():
    local = _caps(Format.Q8_0, Format.TURBO3, Format.TURBO4, Format.FP16)
    remote = _caps(Format.Q8_0, Format.TURBO3, Format.TURBO4, Format.FP16)
    config = negotiate_format(local, remote, override_k=Format.FP16, override_v=Format.TURBO4)
    assert config.fmt_k == Format.FP16
    assert config.fmt_v == Format.TURBO4


def test_negotiate_override_not_available():
    local = _caps(Format.Q8_0, Format.TURBO3)
    remote = _caps(Format.Q8_0, Format.TURBO3)
    with pytest.raises(NegotiationError, match="Override fmt_k"):
        negotiate_format(local, remote, override_k=Format.FP16)


def test_negotiate_respects_limits():
    local = _caps(Format.Q8_0, Format.TURBO3, max_hd=256, max_seq=65536)
    remote = _caps(Format.Q8_0, Format.TURBO3, max_hd=512, max_seq=131072)
    config = negotiate_format(local, remote)
    assert config.max_head_dim == 256
    assert config.max_seq_len == 65536

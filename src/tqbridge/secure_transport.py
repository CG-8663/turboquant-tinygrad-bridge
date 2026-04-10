"""Secure transport layer for TQBridge mesh — TLS 1.3 + mTLS + IPv6.

Every node in the mesh must present a valid certificate signed by
the mesh CA. This prevents unauthorized nodes from joining and
ensures all KV cache data is encrypted in transit.

Setup:
    # Generate mesh CA and node certificates (one time)
    python -m tqbridge.secure_transport --init-ca
    python -m tqbridge.secure_transport --issue-cert node1
    python -m tqbridge.secure_transport --issue-cert node2

    # Start secure decode server
    python -m tqbridge.serve_decode --port 9473 --tls --cert node1.pem --key node1-key.pem --ca ca.pem

    # Connect securely
    sender = SecureSender("node2.mesh.local", 9473, cert="node2.pem", key="node2-key.pem", ca="ca.pem")
"""

from __future__ import annotations

import os
import ssl
import socket
import struct
import time
import datetime
from pathlib import Path
from dataclasses import dataclass

from tqbridge.wire import WireHeader, encode_header


# Default mesh CA directory
MESH_CA_DIR = Path.home() / ".tqbridge" / "pki"


@dataclass
class MeshNode:
    """A node in the secure mesh."""
    name: str
    host: str
    port: int = 9473
    cert: Path | None = None
    key: Path | None = None


class SecureSender:
    """Send compressed KV over TLS 1.3 with mutual authentication.

    Both sender and receiver must present certificates signed by the mesh CA.
    Supports IPv4 and IPv6 addresses.
    """

    def __init__(
        self,
        host: str,
        port: int = 9473,
        cert: str | Path | None = None,
        key: str | Path | None = None,
        ca: str | Path | None = None,
        timeout_s: float = 10.0,
        max_retries: int = 3,
    ):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._sock = None
        self._ssl_sock = None

        # Build TLS context
        self._ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._ctx.minimum_version = ssl.TLSVersion.TLSv1_3

        if ca:
            self._ctx.load_verify_locations(str(ca))
        else:
            ca_path = MESH_CA_DIR / "ca.pem"
            if ca_path.exists():
                self._ctx.load_verify_locations(str(ca_path))

        if cert and key:
            self._ctx.load_cert_chain(str(cert), str(key))
        else:
            # Try default node cert
            cert_path = MESH_CA_DIR / "node.pem"
            key_path = MESH_CA_DIR / "node-key.pem"
            if cert_path.exists() and key_path.exists():
                self._ctx.load_cert_chain(str(cert_path), str(key_path))

        # Verify peer certificate
        self._ctx.verify_mode = ssl.CERT_REQUIRED
        self._ctx.check_hostname = False  # We verify by CA, not hostname

    @property
    def connected(self) -> bool:
        return self._ssl_sock is not None

    def connect(self) -> None:
        """Connect with TLS handshake and mutual authentication."""
        self.close()

        # Resolve address (IPv4 or IPv6)
        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_s)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((self.host, self.port))

            self._ssl_sock = self._ctx.wrap_socket(sock, server_hostname=self.host)
            self._sock = sock

            # Log peer certificate info
            peer_cert = self._ssl_sock.getpeercert()
            if peer_cert:
                subject = dict(x[0] for x in peer_cert.get("subject", ()))
                cn = subject.get("commonName", "unknown")
                print(f"[TLS] Connected to {self.host}:{self.port} — peer: {cn}")

        except (ssl.SSLError, ConnectionRefusedError, TimeoutError, OSError) as e:
            self.close()
            raise ConnectionError(f"Secure connect to {self.host}:{self.port} failed: {e}") from e

    def send_kv(self, k_data: bytes, v_data: bytes, header: WireHeader) -> float:
        """Send KV over encrypted TLS channel."""
        hdr_bytes = encode_header(header)
        payload = hdr_bytes + k_data + v_data
        last_error = None

        for attempt in range(self.max_retries):
            try:
                if self._ssl_sock is None:
                    self.connect()

                t0 = time.perf_counter()
                self._ssl_sock.sendall(payload)
                return (time.perf_counter() - t0) * 1000

            except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError) as e:
                last_error = e
                self.close()
                if attempt < self.max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))

        raise ConnectionError(
            f"Secure send to {self.host}:{self.port} failed after {self.max_retries} "
            f"attempts: {last_error}"
        )

    def close(self):
        if self._ssl_sock:
            try:
                self._ssl_sock.close()
            except OSError:
                pass
            self._ssl_sock = None
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class SecureReceiver:
    """Receive compressed KV over TLS with mutual authentication.

    Requires connecting clients to present a valid certificate.
    Supports IPv4 and IPv6.
    """

    def __init__(
        self,
        port: int = 9473,
        cert: str | Path | None = None,
        key: str | Path | None = None,
        ca: str | Path | None = None,
        ipv6: bool = True,
    ):
        self.port = port
        self.ipv6 = ipv6

        # Build server TLS context
        self._ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._ctx.minimum_version = ssl.TLSVersion.TLSv1_3

        if ca:
            self._ctx.load_verify_locations(str(ca))
        else:
            ca_path = MESH_CA_DIR / "ca.pem"
            if ca_path.exists():
                self._ctx.load_verify_locations(str(ca_path))

        if cert and key:
            self._ctx.load_cert_chain(str(cert), str(key))
        else:
            cert_path = MESH_CA_DIR / "node.pem"
            key_path = MESH_CA_DIR / "node-key.pem"
            if cert_path.exists() and key_path.exists():
                self._ctx.load_cert_chain(str(cert_path), str(key_path))

        # Require client certificate (mTLS)
        self._ctx.verify_mode = ssl.CERT_REQUIRED

    def start(self, on_receive=None):
        """Start listening for secure connections."""
        family = socket.AF_INET6 if self.ipv6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if self.ipv6:
            # Allow both IPv4 and IPv6 connections on IPv6 socket
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)

        sock.bind(("", self.port))
        sock.listen(8)
        sock.settimeout(1.0)

        ssl_sock = self._ctx.wrap_socket(sock, server_side=True)
        print(f"[TLS] Listening on {'[::]' if self.ipv6 else '0.0.0.0'}:{self.port} (TLS 1.3, mTLS)")

        # Accept loop
        while True:
            try:
                conn, addr = ssl_sock.accept()
                peer = dict(x[0] for x in (conn.getpeercert() or {}).get("subject", ()))
                cn = peer.get("commonName", "unknown")
                print(f"[TLS] Accepted from {addr[0]}:{addr[1]} — peer: {cn}")

                if on_receive:
                    self._handle(conn, on_receive)
            except socket.timeout:
                continue
            except ssl.SSLError as e:
                print(f"[TLS] Rejected connection: {e}")

    def _handle(self, conn, on_receive):
        """Handle a single authenticated connection."""
        try:
            while True:
                hdr_data = self._recv_exact(conn, 40)
                if not hdr_data:
                    break

                from tqbridge.wire import decode_header
                header = decode_header(hdr_data)

                payload = self._recv_exact(conn, header.payload_bytes)
                if not payload:
                    break

                mid = len(payload) // 2
                on_receive(header, payload[:mid], payload[mid:])
        finally:
            conn.close()

    def _recv_exact(self, conn, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
                if not chunk:
                    return None
                buf.extend(chunk)
            except (socket.timeout, ssl.SSLError):
                return None
        return bytes(buf)


# ── Certificate Management ──────────────────────────────────────

def init_mesh_ca(ca_dir: Path = MESH_CA_DIR) -> tuple[Path, Path]:
    """Generate a self-signed CA for the mesh.

    Returns (ca_cert_path, ca_key_path).
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError:
        raise RuntimeError("pip install cryptography — needed for certificate generation")

    ca_dir.mkdir(parents=True, exist_ok=True)

    # Generate CA key (ECDSA P-256)
    ca_key = ec.generate_private_key(ec.SECP256R1())

    # Build CA certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "TQBridge Mesh CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Chronara Group"),
    ])

    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # Save
    ca_cert_path = ca_dir / "ca.pem"
    ca_key_path = ca_dir / "ca-key.pem"

    ca_cert_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    ca_key_path.write_bytes(ca_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))

    # Restrict key permissions
    os.chmod(ca_key_path, 0o600)

    print(f"[CA] Created mesh CA at {ca_dir}")
    print(f"  Certificate: {ca_cert_path}")
    print(f"  Private key: {ca_key_path}")
    return ca_cert_path, ca_key_path


def issue_node_cert(node_name: str, ca_dir: Path = MESH_CA_DIR) -> tuple[Path, Path]:
    """Issue a certificate for a mesh node, signed by the CA.

    Returns (cert_path, key_path).
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError:
        raise RuntimeError("pip install cryptography")

    # Load CA
    ca_cert = x509.load_pem_x509_certificate((ca_dir / "ca.pem").read_bytes())
    ca_key = serialization.load_pem_private_key((ca_dir / "ca-key.pem").read_bytes(), password=None)

    # Generate node key
    node_key = ec.generate_private_key(ec.SECP256R1())

    # Build node certificate
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, node_name),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Chronara Mesh"),
    ])

    node_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(node_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(node_name)]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Save
    cert_path = ca_dir / f"{node_name}.pem"
    key_path = ca_dir / f"{node_name}-key.pem"

    cert_path.write_bytes(node_cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(node_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    os.chmod(key_path, 0o600)

    print(f"[CA] Issued certificate for '{node_name}'")
    print(f"  Certificate: {cert_path}")
    print(f"  Private key: {key_path}")
    return cert_path, key_path


# ── CLI ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TQBridge mesh PKI management")
    parser.add_argument("--init-ca", action="store_true", help="Create mesh CA")
    parser.add_argument("--issue-cert", type=str, help="Issue node certificate")
    parser.add_argument("--ca-dir", type=str, default=str(MESH_CA_DIR))
    args = parser.parse_args()

    ca_dir = Path(args.ca_dir)

    if args.init_ca:
        init_mesh_ca(ca_dir)
    elif args.issue_cert:
        issue_node_cert(args.issue_cert, ca_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

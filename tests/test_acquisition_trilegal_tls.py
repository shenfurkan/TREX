"""TRILEGAL TLS completion verifies trust and never retries a submitted form."""

import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trex.acquisition import trilegal_tls
from trex.acquisition.transport import NetworkPolicy
from trex.acquisition.trilegal_errors import TrilegalUnavailableError

ISSUER_URL = "https://crt.sectigo.com/ca.der"


def test_the_post_is_not_retried_but_get_is():
    session = trilegal_tls.create_session(NetworkPolicy(True, attempts=4, backoff_seconds=0))
    retries = session.get_adapter("https://example.org/").max_retries
    assert retries.total == 3 and retries.status_forcelist == [429, 500, 502, 503, 504]
    assert retries.allowed_methods is not None and "POST" not in retries.allowed_methods
    assert "GET" in retries.allowed_methods
    session.close()


def test_a_socket_failure_is_a_retryable_typed_outage(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise ssl.SSLError("handshake failed")

    monkeypatch.setattr(trilegal_tls.ssl, "get_server_certificate", refuse)
    with pytest.raises(TrilegalUnavailableError, match="leaf certificate request failed") as raised:
        trilegal_tls.create_aia_completed_session("example.org", NetworkPolicy(True))
    assert raised.value.retryable
    assert raised.value.failure.category == "transport"


def test_an_unreadable_certifi_bundle_is_a_retryable_typed_outage(monkeypatch):
    monkeypatch.setattr(trilegal_tls.certifi, "where", lambda: "/nonexistent/certifi.pem")
    with pytest.raises(TrilegalUnavailableError, match="certifi root bundle unreadable") as raised:
        trilegal_tls.create_aia_completed_session("example.org", NetworkPolicy(True))
    assert raised.value.retryable


def _leaf_pem():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import AuthorityInformationAccessOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "synthetic leaf")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("example.org")]), critical=False)
        .add_extension(
            x509.AuthorityInformationAccess([
                x509.AccessDescription(
                    AuthorityInformationAccessOID.CA_ISSUERS,
                    x509.UniformResourceIdentifier(ISSUER_URL)),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")


def test_an_unreadable_leaf_or_issuer_payload_is_typed():
    with pytest.raises(TrilegalUnavailableError, match="leaf certificate is unreadable"):
        trilegal_tls.verified_aia_intermediate("not a certificate", lambda _url: b"", b"")
    fetched = []

    def fetch_issuer(url):
        fetched.append(url)
        return b"junk"

    roots = Path(trilegal_tls.certifi.where()).read_bytes()
    with pytest.raises(TrilegalUnavailableError, match="no permitted caIssuers certificate") as raised:
        trilegal_tls.verified_aia_intermediate(_leaf_pem(), fetch_issuer, roots, hostname="example.org")
    assert fetched == [ISSUER_URL]
    assert f"{ISSUER_URL}: ValueError:" in str(raised.value)
    assert raised.value.to_manifest()["category"] == "tls"


def test_an_issuer_payload_over_the_size_bound_is_refused(monkeypatch):
    seen = []
    monkeypatch.setattr(trilegal_tls.certifi, "where", lambda: __file__)

    class Response:
        status_code = 200
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, _size):
            for _ in range(9):
                yield b"x" * 8192

    response = Response()

    def get(url, **kwargs):
        assert kwargs["stream"] is True and kwargs["allow_redirects"] is False
        seen.append(url)
        return response

    monkeypatch.setattr(trilegal_tls.requests, "get", get)

    def verification(leaf, fetch_issuer, _roots, *, hostname):
        assert leaf == "leaf" and hostname == "example.org"
        return fetch_issuer(ISSUER_URL), ISSUER_URL

    monkeypatch.setattr(trilegal_tls, "verified_aia_intermediate", verification)
    monkeypatch.setattr(trilegal_tls.ssl, "get_server_certificate", lambda *_args, **_kwargs: "leaf")
    with pytest.raises(TrilegalUnavailableError, match="exceeds the size bound"):
        trilegal_tls.create_aia_completed_session("example.org", NetworkPolicy(True))
    assert seen == [ISSUER_URL]
    assert response.closed


def test_an_aia_failure_keeps_its_endpoint_and_is_retryable(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("no socket")

    monkeypatch.setattr(trilegal_tls.ssl, "get_server_certificate", refuse)
    with pytest.raises(TrilegalUnavailableError) as raised:
        trilegal_tls.create_aia_completed_session("stev.oapd.inaf.it", NetworkPolicy(True))
    assert raised.value.endpoint == "https://stev.oapd.inaf.it/cgi-bin/trilegal_1.6"
    assert raised.value.to_manifest()["endpoint"] == "stev.oapd.inaf.it"
    assert raised.value.retryable and raised.value.failure.retryable


@pytest.mark.parametrize(
    ("url", "permitted"),
    [
        ("file:///etc/passwd", False),
        ("gopher://169.254.169.254/latest/meta-data/", False),
        ("ftp://sectigo.com/intermediate.crt", False),
        ("http://user@sectigo.com/x.crt", False),
        ("https://sectigo.com:8443/x.crt", False),
        ("http://evil-sectigo.com/x.crt", False),
        ("http://crt.sectigo.com/x.crt", True),
        ("https://sectigo.com/x.crt", True),
    ],
)
def test_caissuers_fetches_are_bounded_to_permitted_ca_publishers(url, permitted):
    """The AIA fetch gate refuses non-HTTP schemes, embedded credentials,
    non-default ports and lookalike hosts before any request is made, and
    returns URLs inside the permitted publisher namespaces unchanged."""
    if permitted:
        assert trilegal_tls._issuer_url(url) == url
    else:
        with pytest.raises(
            TrilegalUnavailableError, match="outside the permitted CA publisher namespaces"
        ):
            trilegal_tls._issuer_url(url)

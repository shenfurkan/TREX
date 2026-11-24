"""Verify a TRILEGAL leaf's missing intermediate against trusted roots.

The TRILEGAL server has served its leaf certificate without the intermediate,
which fails ordinary verification. The fallback fetches that intermediate from
the leaf's Authority Information Access (caIssuers) URL and admits it only if
it is a CA that directly issued the leaf and is directly issued by a certifi
root; verification is never disabled.

Session attributes (shared with an embedding supervisor that builds its own
sessions): ``exonym_ca_source`` names the trust source, and an AIA-completed
session also carries ``exonym_ca_bundle_path`` (a temporary PEM bundle the
caller must delete after use) and ``exonym_aia_intermediate`` (provenance).
These attribute names keep their spelling because the supervisor's session
builder is the producer; renaming them here alone would silently drop its
trust provenance and leak the temporary bundle.
``requests``, ``certifi``, ``urllib3`` and ``cryptography`` are declared
dependencies of TREX.
"""

import os
from pathlib import Path
import ssl
import tempfile
from urllib.parse import urlsplit

import certifi
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .transport import AcquisitionFailure, request_user_agent
from .trilegal_errors import TrilegalUnavailableError

# Largest caIssuers certificate payload accepted [bytes]; a real one is a few kB.
MAX_ISSUER_BYTES = 65536
_ISSUER_DOMAINS = ("sectigo.com", "usertrust.com", "digicert.com", "globalsign.com", "entrust.net",
                   "letsencrypt.org")


def _issuer_url(url):
    """Permit only public CA publisher namespaces; never follow an AIA redirect."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme not in ("http", "https") or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 80 if parsed.scheme == "http" else 443)
            or not any(host == domain or host.endswith("." + domain) for domain in _ISSUER_DOMAINS)):
        raise TrilegalUnavailableError("caIssuers URL is outside the permitted CA publisher namespaces")
    return url


def verified_aia_intermediate(leaf_pem, fetch_issuer, roots_pem, *, hostname=None):
    """Complete a server chain with cryptography's RFC 5280 path verifier.

    The leaf's first DNS SAN is used only when the historical caller omits
    hostname; the subsequent HTTPS handshake still verifies its actual hostname.
    Issuers are bounded, publisher-restricted, and tried in document order.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509.oid import AuthorityInformationAccessOID, ExtensionOID
    from cryptography.x509.verification import PolicyBuilder, Store, VerificationError

    try:
        leaf = x509.load_pem_x509_certificate(leaf_pem.encode("ascii"))
    except (UnicodeError, ValueError) as exc:
        raise TrilegalUnavailableError(f"served leaf certificate is unreadable: {exc}") from exc
    try:
        access = leaf.extensions.get_extension_for_oid(ExtensionOID.AUTHORITY_INFORMATION_ACCESS).value
    except x509.ExtensionNotFound as exc:
        raise TrilegalUnavailableError("leaf certificate has no Authority Information Access extension") from exc
    urls = [description.access_location.value for description in access
            if description.access_method == AuthorityInformationAccessOID.CA_ISSUERS
            and isinstance(description.access_location, x509.UniformResourceIdentifier)]
    if not urls:
        raise TrilegalUnavailableError("leaf certificate lists no caIssuers URL")
    try:
        roots = x509.load_pem_x509_certificates(roots_pem)
        if hostname is None:
            names = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            hostname = names.get_values_for_type(x509.DNSName)[0]
        verifier = PolicyBuilder().store(Store(roots)).build_server_verifier(x509.DNSName(hostname))
    except (ValueError, IndexError, x509.ExtensionNotFound) as exc:
        raise TrilegalUnavailableError("TLS root bundle or leaf DNS identity is invalid") from exc
    failures = []
    for url in dict.fromkeys(urls):
        try:
            _issuer_url(url)
            payload = fetch_issuer(url)
            if not isinstance(payload, bytes) or len(payload) > MAX_ISSUER_BYTES:
                raise TrilegalUnavailableError("caIssuers certificate exceeds the size bound or is not bytes")
            try:
                intermediate = x509.load_der_x509_certificate(payload)
            except ValueError:
                intermediate = x509.load_pem_x509_certificate(payload)
            verifier.verify(leaf, [intermediate])
        except (TrilegalUnavailableError, ValueError, VerificationError, requests.RequestException) as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            continue
        return intermediate.public_bytes(Encoding.PEM), url
    raise TrilegalUnavailableError("no permitted caIssuers certificate completes a trusted server chain: "
                                  + "; ".join(failures), category="tls")


def create_session(policy):
    """Create a verifying session under the caller's explicit network policy."""
    if not policy.allow_network:
        raise TrilegalUnavailableError("network access is disabled by policy")
    session = requests.Session()
    session.headers["User-Agent"] = request_user_agent(policy)
    try:
        bundle = Path(certifi.where())
        session.verify = str(bundle) if bundle.is_file() and bundle.stat().st_size else True
    except Exception:  # A broken certifi installation must retain system verification.
        session.verify = True
    session.exonym_ca_source = "certifi" if session.verify is not True else "system-trust-store"
    # POST is not retried: a resent form would start a second TRILEGAL simulation.
    retry = Retry(total=policy.attempts - 1, backoff_factor=policy.backoff_seconds,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def create_aia_completed_session(host, policy):
    """Complete a chain only with an issuer verified against certifi roots.

    Socket, TLS and issuer-download failures raise a retryable
    ``TrilegalUnavailableError``; the caller deletes ``exonym_ca_bundle_path``.
    """
    if not policy.allow_network:
        raise TrilegalUnavailableError("network access is disabled by policy")
    # The direct leaf probe only selects an issuer; the later request (including
    # through a proxy) still verifies its own chain and hostname against trusted roots.

    def unavailable(what, exc):
        reason = f"{what}: {type(exc).__name__}: {exc}"
        return TrilegalUnavailableError("TRILEGAL AIA completion failed, " + reason, AcquisitionFailure(
            host, "transport", True, 1, reason, None, type(exc).__name__))

    try:
        roots = Path(certifi.where()).read_bytes()
    except OSError as exc:
        raise unavailable("certifi root bundle unreadable", exc) from exc
    try:
        leaf = ssl.get_server_certificate((host, 443), timeout=policy.timeout_seconds)
    except OSError as exc:  # includes ssl.SSLError and socket timeouts
        raise unavailable("leaf certificate request failed", exc) from exc

    def fetch_issuer(url):
        try:
            _issuer_url(url)
            with requests.get(url, timeout=policy.timeout_seconds, stream=True, allow_redirects=False,
                               headers={"User-Agent": request_user_agent(policy)}) as response:
                if 300 <= response.status_code < 400:
                    raise TrilegalUnavailableError("caIssuers redirects are not permitted", category="tls")
                response.raise_for_status()
                payload = b""
                for block in response.iter_content(8192):
                    payload += block
                    if len(payload) > min(MAX_ISSUER_BYTES, policy.max_response_bytes):
                        raise TrilegalUnavailableError("caIssuers certificate exceeds the size bound")
                return payload
        except requests.RequestException as exc:
            raise unavailable("caIssuers download failed", exc) from exc

    intermediate, url = verified_aia_intermediate(leaf, fetch_issuer, roots, hostname=host)
    descriptor, name = tempfile.mkstemp(suffix=".pem")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(roots.rstrip(b"\n") + b"\n" + intermediate)
        session = create_session(policy)
        session.verify = name
        session.exonym_ca_source = "certifi+aia-verified-intermediate"
        session.exonym_ca_bundle_path = name
        session.exonym_aia_intermediate = {"caissuers_url": url}
        return session
    except BaseException:
        os.unlink(name)
        raise

"""TPF acquisition binds target, sectors and raw archive responses."""

import json
from io import BytesIO
from urllib.parse import parse_qs

import pytest
import numpy as np
from astropy.io import fits

from trex.acquisition import transport as network
from trex.acquisition.tpf import acquire_tpf_products, load_tpf
from trex.acquisition.transport import AcquisitionError, AcquisitionFailure


class Transport:
    def __init__(self, observations, products, body=b"raw FITS bytes"):
        self.responses = [observations, products]
        self.transport_failures = []
        self.downloads = []
        self.body = body

    def post_mast(self, _url, data):
        request = json.loads(parse_qs(data.decode())["request"][0])
        assert request["service"] in ("Mast.Caom.Filtered", "Mast.Caom.Products")
        rows = self.responses.pop(0)
        return json.dumps({"status": "COMPLETE", "data": rows,
                           "paging": {"pagesFiltered": 1, "rowsFiltered": len(rows)}}).encode()

    def download(self, url, path):
        self.downloads.append(url)
        path.write_bytes(self.body)
        return path


def _observation(obsid, exposure, sector=1):
    return {"obsid": str(obsid), "target_name": "123", "project": "TESS", "provenance_name": "SPOC",
            "sequence_number": sector, "t_exptime": exposure}


def _product(obsid, name):
    return {"parent_obsid": str(obsid), "productFilename": name, "dataURI": "mast:TESS/product/" + name}


def _fits_bytes(tmp_path, *, tic=123, sector=1, cadences=50):
    primary = fits.PrimaryHDU()
    primary.header.update(TELESCOP="TESS", TICID=tic, SECTOR=sector, CAMERA=1, CCD=2)
    pixels = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.arange(cadences, dtype=float)),
        fits.Column(name="QUALITY", format="J", array=np.zeros(cadences, dtype=int)),
        fits.Column(name="FLUX", format="4E", dim="(2,2)", unit="e-/s", array=np.ones((cadences, 2, 2))),
    ])
    pixels.header.update(TIMESYS="TDB", TIMEUNIT="d", BJDREFI=2457000, BJDREFF=0.)
    path = tmp_path / f"fixture_{tic}_{sector}_{cadences}.fits"
    fits.HDUList([primary, pixels, fits.ImageHDU(np.full((2, 2), 2))]).writeto(path, overwrite=True, checksum=True)
    return path.read_bytes()


def _table(rows):
    return json.dumps({"status": "COMPLETE", "data": rows,
                       "paging": {"pagesFiltered": 1, "rowsFiltered": len(rows)}}).encode()


def _install_responses(monkeypatch, outcomes):
    outcomes = iter(outcomes)
    calls = []

    class Response(BytesIO):
        headers = {}

    def respond(request, timeout):
        calls.append((request.full_url, timeout))
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return Response(outcome)

    monkeypatch.setattr(network, "urlopen", respond)
    monkeypatch.setattr(network, "sleep_before_retry", lambda _delay: None)
    return calls


def test_tpf_selects_shortest_declared_cadence_and_retains_responses(tmp_path):
    observations = [_observation(1, 120), _observation(2, 20)]
    products = [_product(1, "long_tp.fits"), _product(2, "short_fast-tp.fits")]
    body = _fits_bytes(tmp_path)
    transport = Transport(observations, products, body)
    records = acquire_tpf_products("123", [1], tmp_path, transport)
    assert records[0]["cadence_seconds"] == 20
    assert (tmp_path / records[0]["path"]).read_bytes() == body
    assert len(transport.downloads) == 1 and transport.downloads[0].endswith("short_fast-tp.fits")
    assert len(list(tmp_path.glob("mast_response_*.json"))) == 2
    assert json.loads((tmp_path / "tpf_products.json").read_text())["status"] == "available"


def test_tpf_refuses_ambiguous_shortest_products(tmp_path):
    transport = Transport([_observation(1, 120)], [_product(1, "a_tp.fits"), _product(1, "b_tp.fits")])
    with pytest.raises(AcquisitionError, match="ambiguous shortest"):
        acquire_tpf_products("123", [1], tmp_path, transport)
    assert not transport.downloads


def test_invalid_download_is_retained_but_never_published_available(tmp_path):
    transport = Transport([_observation(1, 120)], [_product(1, "a_tp.fits")])
    with pytest.raises(AcquisitionError, match="invalid TESS TPF"):
        acquire_tpf_products("123", [1], tmp_path, transport)
    record = json.loads((tmp_path / "tpf_products.json").read_text())
    assert record["status"] == "unavailable"
    assert (tmp_path / record["failed_product"]["path"]).read_bytes() == b"raw FITS bytes"


def test_access_denial_on_tpf_download_is_recorded_without_fallback(tmp_path):
    transport = Transport([_observation(1, 120)], [_product(1, "a_tp.fits")])

    def denied(_url, _path):
        transport.transport_failures.append(AcquisitionFailure("mast.stsci.edu", "http_status", False, 1,
                                                                "HTTP 403", 403, "HTTPError"))
        return None

    transport.download = denied
    with pytest.raises(AcquisitionError, match="TPF download failed") as exc:
        acquire_tpf_products("123", [1], tmp_path, transport)
    assert not exc.value.retryable
    record = json.loads((tmp_path / "tpf_products.json").read_text())
    assert record["status"] == "unavailable" and record["network_failure"]["http_status"] == 403


def test_download_failure_does_not_reuse_a_stale_network_failure(tmp_path):
    service = Transport([_observation(1, 120)], [_product(1, "a_tp.fits")])
    service.transport_failures.append(AcquisitionFailure("old.example", "http_status", False, 1, "old denial", 403))
    service.download = lambda *_args: None
    with pytest.raises(AcquisitionError, match="TPF download failed") as caught:
        acquire_tpf_products("123", [1], tmp_path, service)
    assert caught.value.network_failure is None and not caught.value.retryable
    assert "network_failure" not in json.loads((tmp_path / "tpf_products.json").read_text())


def test_tpf_loader_preserves_btjd_units_and_quality(tmp_path):
    primary = fits.PrimaryHDU()
    primary.header.update(TELESCOP="TESS", TICID=123, SECTOR=1, CAMERA=1, CCD=2)
    pixels = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.arange(1., 51.) * 86400.),
        fits.Column(name="QUALITY", format="J", array=[0, 1] * 25),
        fits.Column(name="FLUX", format="4E", dim="(2,2)", unit="e-/s", array=np.ones((50, 2, 2))),
    ])
    pixels.header.update(TIMESYS="TDB", TIMEUNIT="s", BJDREFI=2457001, BJDREFF=0.)
    path = tmp_path / "tpf.fits"
    fits.HDUList([primary, pixels, fits.ImageHDU(np.full((2, 2), 2))]).writeto(path)
    result = load_tpf(path, "123", 1)
    np.testing.assert_array_equal(result["time"], np.arange(2., 52.))
    np.testing.assert_array_equal(result["quality_retained"], [True, False] * 25)
    assert result["time_system"] == "BTJD_TDB" and result["flux_unit"] == "e-/s"
    with pytest.raises(AcquisitionError, match="TIC identity"):
        load_tpf(path, "124", 1)
    pixels.header["TIMESYS"] = "UTC"
    fits.HDUList([primary, pixels, fits.ImageHDU(np.full((2, 2), 2))]).writeto(path, overwrite=True)
    with pytest.raises(AcquisitionError, match="TIMESYS"):
        load_tpf(path, "123", 1)


def test_loader_refuses_an_empty_or_mis_shaped_spoc_aperture(tmp_path):
    for aperture in (np.zeros((2, 2), dtype=int), np.full((1, 1), 2, dtype=int)):
        primary = fits.PrimaryHDU()
        primary.header.update(TELESCOP="TESS", TICID=123, SECTOR=1, CAMERA=1, CCD=2)
        pixels = fits.BinTableHDU.from_columns([
            fits.Column(name="TIME", format="D", array=np.arange(50, dtype=float)),
            fits.Column(name="QUALITY", format="J", array=np.zeros(50, dtype=int)),
            fits.Column(name="FLUX", format="4E", dim="(2,2)", unit="e-/s", array=np.ones((50, 2, 2))),
        ])
        pixels.header.update(TIMESYS="TDB", TIMEUNIT="d", BJDREFI=2457000)
        path = tmp_path / "tpf.fits"
        fits.HDUList([primary, pixels, fits.ImageHDU(aperture)]).writeto(path, overwrite=True)
        with pytest.raises(AcquisitionError, match="aperture mask is unusable"):
            load_tpf(path, "123", 1)


def _tpf_with_reference(tmp_path, **keywords):
    primary = fits.PrimaryHDU()
    primary.header.update(TELESCOP="TESS", TICID=123, SECTOR=1, CAMERA=1, CCD=2)
    pixels = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.arange(50, dtype=float)),
        fits.Column(name="QUALITY", format="J", array=np.zeros(50, dtype=int)),
        fits.Column(name="FLUX", format="4E", dim="(2,2)", unit="e-/s", array=np.ones((50, 2, 2))),
    ])
    pixels.header.update(TIMESYS="TDB", TIMEUNIT="d", **keywords)
    path = tmp_path / "tpf.fits"
    fits.HDUList([primary, pixels, fits.ImageHDU(np.full((2, 2), 2))]).writeto(path, overwrite=True)
    return path


def test_loader_refuses_a_fractional_reference_without_an_integer_part(tmp_path):
    path = _tpf_with_reference(tmp_path, BJDREFF=0.5)
    with pytest.raises(AcquisitionError, match="BJDREFF without BJDREFI"):
        load_tpf(path, "123", 1)


def test_loader_refuses_mjd_references(tmp_path):
    path = _tpf_with_reference(tmp_path, BJDREFI=2457000, MJDREF=57000.0)
    with pytest.raises(AcquisitionError, match="MJDREF"):
        load_tpf(path, "123", 1)


def test_loader_preserves_split_reference_below_full_jd_precision(tmp_path):
    fraction = 2**-34
    path = _tpf_with_reference(tmp_path, BJDREFI=2457000, BJDREFF=fraction)
    stored_fraction = fits.getheader(path, 1)["BJDREFF"]
    # FITS decimal serialization can move this binary fraction by one ULP;
    # the loader must retain the stored fraction, not the pre-write Python one.
    assert stored_fraction == pytest.approx(fraction, rel=np.finfo(float).eps, abs=0.)
    actual = load_tpf(path, "123", 1)
    np.testing.assert_array_equal(actual["time"], np.arange(50, dtype=float) + stored_fraction)
    assert actual["header"]["BJDREFF"] == stored_fraction


def test_loader_keeps_precise_split_when_full_reference_is_also_present(tmp_path):
    fraction = 2**-34
    path = _tpf_with_reference(tmp_path, BJDREF=2457000., BJDREFI=2457000, BJDREFF=fraction)
    stored_fraction = fits.getheader(path, 1)["BJDREFF"]
    np.testing.assert_array_equal(load_tpf(path, "123", 1)["time"], np.arange(50, dtype=float) + stored_fraction)


def test_loader_preserves_raw_invalid_time_rows_and_quality(tmp_path):
    path = _tpf_with_reference(tmp_path, BJDREFI=2457000, BJDREFF=2**-34)
    with fits.open(path, mode="update") as hdul:
        hdul[1].data["TIME"][0] = np.nan
        hdul[1].data["TIME"][1] = np.inf
        hdul[1].data["QUALITY"][0] = 1
    actual = load_tpf(path, "123", 1)
    assert actual["time"].size == 50 and np.isnan(actual["time"][0]) and np.isposinf(actual["time"][1])
    assert actual["quality"][0] == 1
    np.testing.assert_array_equal(actual["time"][2:], np.arange(2., 50.) + 2**-34)


@pytest.mark.parametrize("keywords", [
    {}, {"BJDREF": 2457001., "BJDREFI": 2457000},
    {"BJDREFI": 2457000.5}, {"BJDREFI": True}, {"BJDREFI": 2457000, "BJDREFF": True},
])
def test_loader_refuses_missing_conflicting_or_non_numeric_references(tmp_path, keywords):
    path = _tpf_with_reference(tmp_path, **keywords)
    with pytest.raises(AcquisitionError, match="BJD"):
        load_tpf(path, "123", 1)


def test_loader_refuses_negative_or_absolute_quality_and_time(tmp_path):
    primary = fits.PrimaryHDU()
    primary.header.update(TELESCOP="TESS", TICID=123, SECTOR=1, CAMERA=1, CCD=2)
    pixels = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.arange(50, dtype=float)),
        fits.Column(name="QUALITY", format="J", array=np.array([-1] + [0] * 49)),
        fits.Column(name="FLUX", format="4E", dim="(2,2)", unit="e-/s", array=np.ones((50, 2, 2))),
    ])
    pixels.header.update(TIMESYS="TDB", TIMEUNIT="d", BJDREFI=2457000)
    path = tmp_path / "tpf.fits"
    fits.HDUList([primary, pixels, fits.ImageHDU(np.full((2, 2), 2))]).writeto(path)
    with pytest.raises(AcquisitionError, match="nonnegative integer"):
        load_tpf(path, "123", 1)

    primary = fits.PrimaryHDU()
    primary.header.update(TELESCOP="TESS", TICID=123, SECTOR=1, CAMERA=1, CCD=2)
    pixels = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.full(50, 2457001.0)),
        fits.Column(name="QUALITY", format="J", array=np.zeros(50, dtype=int)),
        fits.Column(name="FLUX", format="4E", dim="(2,2)", unit="e-/s", array=np.ones((50, 2, 2))),
    ])
    pixels.header.update(TIMESYS="TDB", TIMEUNIT="d", BJDREFI=2457000)
    fits.HDUList([primary, pixels, fits.ImageHDU(np.full((2, 2), 2))]).writeto(path, overwrite=True)
    with pytest.raises(AcquisitionError, match="absolute dates"):
        load_tpf(path, "123", 1)


def test_real_transport_recovers_from_mast_error_and_retains_each_response(tmp_path, monkeypatch):
    bad = b'{"status":"ERROR","msg":"MAST service temporarily busy"}'
    observations = _table([_observation(1, 120)])
    products = _table([_product(1, "a_tp.fits")])
    fits_bytes = _fits_bytes(tmp_path)
    calls = _install_responses(monkeypatch, [bad, observations, products, fits_bytes])
    service = network.AcquisitionTransport(network.NetworkPolicy(True, attempts=3, timeout_seconds=9))
    output = tmp_path / "acquisition"

    records = acquire_tpf_products("123", [1], output, service)

    assert len(calls) == 4 and all(timeout == 9 for _, timeout in calls)
    assert (output / records[0]["path"]).read_bytes() == fits_bytes
    manifest = json.loads((output / "tpf_products.json").read_text())
    query = manifest["observations"]
    assert query["status"] == "available"
    first, second = query["attempt_history"]
    assert first["failure"]["category"] == "provider_error"
    assert first["failure"]["event"] == "mast.provider_error"
    assert first["failure"]["message"] == "MAST service temporarily busy"
    assert first["recorded_utc"].endswith("Z") and second["recorded_utc"].endswith("Z")
    assert (output / first["response"]).read_bytes() == bad
    assert (output / second["response"]).read_bytes() == observations


@pytest.mark.parametrize("key", ["msg", "message", "reason", "detail", "error"])
def test_provider_error_reason_is_retained_without_a_data_table(tmp_path, monkeypatch, key):
    body = json.dumps({"status": "ERROR", key: "provider detail"}).encode()
    calls = _install_responses(monkeypatch, [body])
    service = network.AcquisitionTransport(network.NetworkPolicy(True, attempts=1))
    with pytest.raises(AcquisitionError, match="provider detail") as caught:
        acquire_tpf_products("123", [1], tmp_path, service)
    record = json.loads((tmp_path / "tpf_products.json").read_text())
    assert len(calls) == 1 and caught.value.retryable and record["category"] == "provider_error"
    assert "provider detail" in record["reason"]
    assert (tmp_path / record["failed_query"]["response"]).read_bytes() == body


@pytest.mark.parametrize("outcomes,category,reason", [
    ([b'{"status":"ERROR","msg":"busy 1"}', b'{"status":"ERROR","msg":"busy 2"}',
      b'{"status":"ERROR","msg":"busy 3"}'], "provider_error", "busy 3"),
    ([b'{"status":"ERROR","msg":"busy 1"}', TimeoutError("timeout 2"),
      TimeoutError("timeout 3")], "transport_failure", "timeout 3"),
])
def test_exhaustion_records_provider_or_transport_cause_and_raw_history(tmp_path, monkeypatch, outcomes, category, reason):
    calls = _install_responses(monkeypatch, outcomes)
    service = network.AcquisitionTransport(network.NetworkPolicy(True, attempts=3))
    with pytest.raises(AcquisitionError, match=reason) as caught:
        acquire_tpf_products("123", [1], tmp_path, service)
    assert caught.value.retryable and len(calls) == 3
    manifest = json.loads((tmp_path / "tpf_products.json").read_text())
    query = manifest["failed_query"]
    assert manifest["status"] == "unavailable" and manifest["category"] == category
    assert reason in manifest["reason"] and len(query["attempt_history"]) == 3
    first = query["attempt_history"][0]
    assert first["failure"]["category"] == "provider_error"
    assert (tmp_path / first["response"]).read_bytes() == outcomes[0]
    if category == "provider_error":
        last = query["attempt_history"][-1]
        assert (tmp_path / last["response"]).read_bytes() == outcomes[-1]
        assert last["failure"]["http_status"] == 200 and last["failure"]["attempts"] == 3
    else:
        assert manifest["network_failure"]["category"] == "connect_timeout"
        assert manifest["network_failure"]["attempts"] == 3


def test_retry_after_exhaustion_keeps_failed_record_and_recovers(tmp_path, monkeypatch):
    bad = b'{"status":"ERROR","msg":"MAST temporarily unavailable"}'
    _install_responses(monkeypatch, [bad, bad])
    service = network.AcquisitionTransport(network.NetworkPolicy(True, attempts=2))
    output = tmp_path / "acquisition"
    with pytest.raises(AcquisitionError, match="temporarily unavailable"):
        acquire_tpf_products("123", [1], output, service)
    failed = json.loads((output / "tpf_products.json").read_text())

    fits_bytes = _fits_bytes(tmp_path)
    _install_responses(monkeypatch, [_table([_observation(1, 120)]), _table([_product(1, "a_tp.fits")]), fits_bytes])
    records = acquire_tpf_products("123", [1], output, service)
    recovered = json.loads((output / "tpf_products.json").read_text())
    assert recovered["status"] == "available" and (output / records[0]["path"]).read_bytes() == fits_bytes
    history = json.loads((output / recovered["previous_record"]).read_text())
    assert history == failed and history["category"] == "provider_error"
    for attempt in history["failed_query"]["attempt_history"]:
        assert (output / attempt["response"]).read_bytes() == bad


def test_provider_error_then_malformed_response_keeps_both_without_retrying_malformed(tmp_path, monkeypatch):
    bad = b'{"status":"ERROR","msg":"busy"}'
    malformed = b"not JSON"
    calls = _install_responses(monkeypatch, [bad, malformed])
    service = network.AcquisitionTransport(network.NetworkPolicy(True, attempts=3))
    with pytest.raises(AcquisitionError, match="invalid"):
        acquire_tpf_products("123", [1], tmp_path, service)
    record = json.loads((tmp_path / "tpf_products.json").read_text())
    attempts = record["failed_query"]["attempt_history"]
    assert len(calls) == 2 and record["category"] == "malformed_response"
    assert (tmp_path / attempts[0]["response"]).read_bytes() == bad
    assert (tmp_path / attempts[1]["response"]).read_bytes() == malformed


@pytest.mark.parametrize("body,category,reason", [
    (b"\xff", "malformed_response", "UTF-8"),
    (b"[1, 2]", "malformed_response", "no data table"),
    # Depending on the interpreter's recursion limit this either fails decoding or lacks a table.
    pytest.param(b"[" * 2000 + b"]" * 2000, "malformed_response", "UTF-8 JSON|no data table", id="deep-json"),
    (b'{"status":"COMPLETE","data":[],"paging":[]}', "malformed_response", "malformed paging"),
    (b'{"status":"COMPLETE","data":[{}],"paging":{"rowsFiltered":2}}',
     "malformed_response", "incomplete or paginated"),
    (b'{"status":"COMPLETE","data":[]}', "empty_result", "no SPOC observations"),
])
def test_malformed_and_successful_empty_responses_are_distinct(tmp_path, monkeypatch, body, category, reason):
    calls = _install_responses(monkeypatch, [body])
    service = network.AcquisitionTransport(network.NetworkPolicy(True, attempts=3))
    with pytest.raises(AcquisitionError, match=reason):
        acquire_tpf_products("123", [1], tmp_path, service)
    manifest = json.loads((tmp_path / "tpf_products.json").read_text())
    assert len(calls) == 1 and manifest["status"] == "unavailable" and manifest["category"] == category
    query = manifest["observations"] if category == "empty_result" else manifest["failed_query"]
    assert (tmp_path / query["response"]).read_bytes() == body
    assert query["status"] == ("available" if category == "empty_result" else "unavailable")


def test_empty_product_table_is_recorded_as_a_successful_query(tmp_path):
    service = Transport([_observation(1, 120)], [])
    with pytest.raises(AcquisitionError, match="no SPOC TPF for sector 1"):
        acquire_tpf_products("123", [1], tmp_path, service)
    record = json.loads((tmp_path / "tpf_products.json").read_text())
    assert record["category"] == "empty_result" and record["product_query"]["status"] == "available"
    assert json.loads((tmp_path / record["product_query"]["response"]).read_text())["data"] == []


class FitsTransport(Transport):
    def __init__(self, observations, products, files):
        super().__init__(observations, products)
        self.files = files

    def download(self, url, path):
        self.downloads.append(url)
        path.write_bytes(self.files[url.rsplit("/", 1)[-1]])
        return path


class RetainedTransport:
    def __init__(self, files=None):
        self.files = files
        self.downloads = []
        self.transport_failures = []

    def post_mast(self, *_args):
        pytest.fail("complete retained MAST responses should be reused")

    def download(self, url, path):
        self.downloads.append(url)
        if self.files is None:
            self.transport_failures.append(AcquisitionFailure("mast.stsci.edu", "connect_timeout", True, 1,
                                                               "replacement timed out", exception_class="TimeoutError"))
            return None
        path.write_bytes(self.files[url.rsplit("/", 1)[-1]])
        return path


def _acquire_two_sectors(tmp_path):
    files = {"one_tp.fits": _fits_bytes(tmp_path, sector=1), "two_tp.fits": _fits_bytes(tmp_path, sector=2)}
    observations = [_observation(1, 120, sector=1), _observation(2, 120, sector=2)]
    products = [_product(1, "one_tp.fits"), _product(2, "two_tp.fits")]
    output = tmp_path / "acquisition"
    records = acquire_tpf_products("123", [1, 2], output, FitsTransport(observations, products, files))
    return output, records, files


def test_retained_products_reuse_verified_queries_and_fits_without_network(tmp_path):
    output, records, files = _acquire_two_sectors(tmp_path)
    original_manifest = (output / "tpf_products.json").read_bytes()
    offline = RetainedTransport()
    assert acquire_tpf_products("123", [1, 2], output, offline) == records
    assert not offline.downloads and (output / "tpf_products.json").read_bytes() == original_manifest
    assert (output / records[0]["path"]).read_bytes() == files["one_tp.fits"]
    assert (output / records[1]["path"]).read_bytes() == files["two_tp.fits"]


@pytest.mark.parametrize("damage", ["checksum", "truncated", "missing", "wrong_tic", "wrong_sector", "wrong_uri"])
def test_recovery_rejects_damaged_or_mismatched_retained_product_and_preserves_good_bytes(tmp_path, damage):
    output, records, files = _acquire_two_sectors(tmp_path)
    first, second = (output / item["path"] for item in records)
    manifest_path = output / "tpf_products.json"
    if damage == "checksum":
        with fits.open(second) as hdul:
            offset = hdul[1].fileinfo()["datLoc"]
        broken = bytearray(second.read_bytes())
        broken[offset] ^= 1
        second.write_bytes(broken)
    elif damage == "truncated":
        second.write_bytes(second.read_bytes()[:100])
    elif damage == "missing":
        second.unlink()
    elif damage == "wrong_tic":
        second.write_bytes(_fits_bytes(tmp_path, tic=999, sector=2))
    elif damage == "wrong_sector":
        second.write_bytes(_fits_bytes(tmp_path, sector=3))
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["products"][1]["uri"] += "-different"
        manifest_path.write_text(json.dumps(manifest))
    damaged_bytes = second.read_bytes() if second.exists() else None

    failed = RetainedTransport()
    with pytest.raises(AcquisitionError, match="replacement timed out"):
        acquire_tpf_products("123", [1, 2], output, failed)
    failure_record = json.loads(manifest_path.read_text())
    assert failure_record["status"] == "unavailable"
    assert failure_record["network_failure"]["category"] == "connect_timeout"
    assert failure_record["products"] == [records[0]] and len(failed.downloads) == 1
    assert first.read_bytes() == files["one_tp.fits"]
    if damaged_bytes is not None:
        assert second.read_bytes() == damaged_bytes
    assert any(item["event"] == "rejected_product" for item in failure_record["recovery_history"])

    recovered = RetainedTransport(files)
    new_records = acquire_tpf_products("123", [1, 2], output, recovered)
    final = json.loads(manifest_path.read_text())
    assert final["status"] == "available" and len(recovered.downloads) == 1
    assert new_records[0] == records[0] and new_records[1]["path"] != records[1]["path"]
    assert (output / new_records[1]["path"]).read_bytes() == files["two_tp.fits"]
    assert first.read_bytes() == files["one_tp.fits"]
    if damaged_bytes is not None:
        assert second.read_bytes() == damaged_bytes
    previous = json.loads((output / final["previous_record"]).read_text())
    assert previous["status"] == "unavailable" and previous["reason"] == failure_record["reason"]
    assert json.loads((output / previous["previous_record"]).read_text())["status"] == "available"


def test_retained_response_identity_and_completeness_are_rechecked(tmp_path):
    output, records, files = _acquire_two_sectors(tmp_path)
    manifest_path = output / "tpf_products.json"
    manifest = json.loads(manifest_path.read_text())
    response = output / manifest["observations"]["response"]
    changed = json.loads(response.read_text())
    changed["data"][0]["target_name"] = "999"
    response.write_text(json.dumps(changed))
    with pytest.raises(AcquisitionError, match="outside the requested target"):
        acquire_tpf_products("123", [1, 2], output, RetainedTransport())
    assert json.loads(manifest_path.read_text())["status"] == "unavailable"
    assert (output / records[0]["path"]).read_bytes() == files["one_tp.fits"]


def test_incomplete_retained_query_is_reacquired_before_reusing_products(tmp_path):
    output, records, files = _acquire_two_sectors(tmp_path)
    manifest_path = output / "tpf_products.json"
    original = json.loads(manifest_path.read_text())
    response = output / original["observations"]["response"]
    changed = json.loads(response.read_text())
    changed["paging"]["rowsFiltered"] += 1
    response.write_text(json.dumps(changed))
    observations = [_observation(1, 120, sector=1), _observation(2, 120, sector=2)]
    service = FitsTransport(observations, [], files)

    assert acquire_tpf_products("123", [1, 2], output, service) == records
    assert not service.downloads and len(service.responses) == 1
    result = json.loads(manifest_path.read_text())
    assert result["observations"]["response"] != original["observations"]["response"]
    assert any(item["event"] == "reacquired_observations" for item in result["recovery_history"])
    assert json.loads(response.read_text())["paging"]["rowsFiltered"] == 3


@pytest.mark.parametrize("change", ["missing_sector", "duplicate_sector", "foreign_sector", "noninteger_sector"])
def test_incomplete_or_duplicated_retained_sector_list_cannot_be_available(tmp_path, change):
    output, records, files = _acquire_two_sectors(tmp_path)
    manifest_path = output / "tpf_products.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "missing_sector":
        manifest["products"].pop()
    elif change == "duplicate_sector":
        manifest["products"].append(dict(manifest["products"][1]))
    elif change == "foreign_sector":
        manifest["products"][1]["sector"] = 3
    else:
        manifest["products"][1]["sector"] = 2.0
    manifest_path.write_text(json.dumps(manifest))
    service = RetainedTransport()

    with pytest.raises(AcquisitionError, match="replacement timed out"):
        acquire_tpf_products("123", [1, 2], output, service)
    current = json.loads(manifest_path.read_text())
    assert current["status"] == "unavailable" and current["products"] == [records[0]]
    assert (output / records[0]["path"]).read_bytes() == files["one_tp.fits"]
    assert (output / records[1]["path"]).read_bytes() == files["two_tp.fits"]

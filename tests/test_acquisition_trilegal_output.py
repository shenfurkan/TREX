"""A TRILEGAL population is cached only when the loader can read every star."""

import pytest

from trex.acquisition.trilegal_errors import TRILEGAL_ENDPOINT, TrilegalUnavailableError
from trex.acquisition.trilegal_output import trilegal_output_to_csv

BODY = "TESS Mact logg logTe [M/H]\n12 1 4.4 3.76 -0.1\n#TRILEGAL normally terminated\n"


def test_population_keeps_cell_text_and_drops_only_the_footer():
    csv, count = trilegal_output_to_csv(BODY)
    assert count == 1 and csv == "TESS,Mact,logg,logTe,[M/H]\n12,1,4.4,3.76,-0.1\n"


@pytest.mark.parametrize("body, message", [
    (BODY.split("#")[0], "termination"),
    (BODY.replace("12 1", "nan 1"), "non-finite"),
    (BODY.replace(" [M/H]", "").replace(" -0.1", ""), r"\[M/H\]"),
    (BODY.replace("-0.1", "x"), r"non-numeric \[M/H\]"),
    (BODY + "#TRILEGAL normally terminated\n", "repeats"),
    (BODY.replace("3.76", "3.76,1"), "comma"),
    ("TESS Mact logg logTe [M/H]\n#TRILEGAL normally terminated\n", "no simulated stars"),
])
def test_population_refuses_what_the_loader_could_not_read(body, message):
    with pytest.raises(TrilegalUnavailableError, match=message):
        trilegal_output_to_csv(body)


def test_refusal_without_transport_failure_names_its_own_provider():
    default = TrilegalUnavailableError("busy", retryable=True).to_manifest()
    assert default["endpoint"] == TRILEGAL_ENDPOINT and default["retryable"] is True
    dust = TrilegalUnavailableError("no map", endpoint="https://irsa.ipac.caltech.edu", category="provider")
    record = dust.to_manifest()
    assert record["endpoint"] == "https://irsa.ipac.caltech.edu" and record["retryable"] is False
    assert record["exception_class"] == "TrilegalUnavailableError" and record["diagnostic_chain"] == ["no map"]

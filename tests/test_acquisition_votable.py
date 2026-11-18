"""Complete TAP responses retain exact identifiers and reject overflow."""

import pytest

from trex.acquisition.votable import parse_votable_bytes


@pytest.mark.parametrize("status", ["OK", "OVERFLOW", "ERROR"])
def test_votable_requires_complete_result(status):
    payload = f'''<?xml version="1.0"?>
    <VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
      <RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/>
        <TABLE><FIELD name="source_id" datatype="long"/>
          <DATA><TABLEDATA><TR><TD>9000000000000000001</TD></TR></TABLEDATA></DATA>
        </TABLE><INFO name="QUERY_STATUS" value="{status}"/>
      </RESOURCE>
    </VOTABLE>'''.encode()
    if status == "OK":
        assert parse_votable_bytes(payload)["data"] == [[9000000000000000001]]
    else:
        with pytest.raises(ValueError, match="not complete"):
            parse_votable_bytes(payload)


def _votable(fields, rows, infos=""):
    header = "".join(f'<FIELD name="{name}" datatype="{kind}"/>' for name, kind in fields)
    body = "".join("<TR>" + "".join(f"<TD>{cell}</TD>" for cell in row) + "</TR>" for row in rows)
    return f'''<?xml version="1.0"?>
    <VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
      <RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/>{infos}
        <TABLE>{header}<DATA><TABLEDATA>{body}</TABLEDATA></DATA></TABLE>
      </RESOURCE>
    </VOTABLE>'''.encode()


def test_masked_cell_becomes_none_and_identifier_stays_exact():
    payload = _votable([("source_id", "long"), ("parallax", "double")], [["9000000000000000001", ""]])
    assert parse_votable_bytes(payload)["data"] == [[9000000000000000001, None]]


def test_truncated_vizier_match_count_is_refused_with_both_counts():
    payload = _votable([("source_id", "long")], [["1"]], '<INFO name="matches" value="2"/>')
    with pytest.raises(ValueError, match="2 matching rows but the response has 1"):
        parse_votable_bytes(payload)


def test_schema_warning_is_refused_with_its_message():
    payload = _votable([("source_id", "long")], [["1"]], '<INFO name="Warning" value="x">column renamed</INFO>')
    with pytest.raises(ValueError, match="column renamed"):
        parse_votable_bytes(payload)


def test_repeated_field_name_is_refused():
    payload = _votable([("source_id", "long"), ("source_id", "long")], [["1", "2"]])
    with pytest.raises(ValueError, match="repeats a field name"):
        parse_votable_bytes(payload)

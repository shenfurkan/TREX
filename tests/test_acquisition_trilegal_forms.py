"""TRILEGAL form controls are checked before query submission."""

import pytest

from trex.acquisition.trilegal_errors import TrilegalUnavailableError
from trex.acquisition.trilegal_forms import LinkParser, submitted_form


def test_form_submits_only_the_first_button_and_validates_choices():
    page = ('<form action="submit"><select name="gal_coord"><option value="1">one</option>'
            '<option value="2" selected>two</option></select><input type="submit" name="submit" value="Go">'
            '<input type="submit" name="reset" value="Reset"></form>')
    action, submitted, encoding = submitted_form(page, {"gal_coord": "2"})
    assert action == "submit" and encoding == "application/x-www-form-urlencoded"
    assert submitted == [("gal_coord", "2"), ("submit", "Go")]
    with pytest.raises(TrilegalUnavailableError, match="does not offer"):
        submitted_form(page, {"gal_coord": "3"})


def test_checkbox_radio_textarea_and_multipart_controls_are_submitted_as_a_browser_would():
    page = ('<form action="/go" enctype="multipart/form-data">'
            '<input type="checkbox" name="a" value="1" checked>'
            '<input type="checkbox" name="b" value="2">'
            '<input type="radio" name="r" value="first"><input type="radio" name="r" value="second" checked>'
            '<input type="text" name="t" value="x"><textarea name="memo">hello</textarea>'
            '<input type="submit" name="send" value="Go"></form>')
    action, fields, encoding = submitted_form(page, {"t": "y"})
    assert action == "/go" and encoding == "multipart/form-data"
    assert fields == [("a", "1"), ("r", "second"), ("t", "y"), ("memo", "hello"), ("send", "Go")]


@pytest.mark.parametrize("page", [
    '<form><select name="gal_coord"><option value="1">one</form>',
    '<form><textarea name="memo">unterminated</form>',
])
def test_a_page_ending_inside_a_select_or_textarea_is_refused(page):
    with pytest.raises(TrilegalUnavailableError, match="ends inside"):
        submitted_form(page, {})


def test_link_parser_keeps_every_href_in_document_order():
    parser = LinkParser()
    parser.feed('<a href="first">1</a><p>no link</p><a>none</a><a href="second">2</a>')
    parser.close()
    assert parser.hrefs == ["first", "second"]

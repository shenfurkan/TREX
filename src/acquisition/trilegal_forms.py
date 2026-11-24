"""TRILEGAL HTML form parsing: the submitted form fields and the result link."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Dict, List, Mapping, Optional, Tuple

from .trilegal_errors import TrilegalUnavailableError

class _TrilegalFormParser(HTMLParser):
    """Collect the first form's successful controls as a browser submits them.

    Mirrors MechanicalSoup's request construction, which the reference
    implementation relies on: unchecked radio/checkbox inputs are omitted, a
    ``select`` submits its selected option or else its first option, and a
    ``textarea`` submits its text.  The offered radio/checkbox and select
    values are retained so an override can be checked against the form.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action: Optional[str] = None
        self.enctype = "application/x-www-form-urlencoded"
        self.fields: List[Tuple[str, str]] = []
        self.offered_values: Dict[str, List[str]] = {}
        self._form_depth = 0
        self._form_done = False
        self._select: Optional[str] = None
        self._select_first: Optional[str] = None
        self._select_chosen: Optional[str] = None
        self._option_attrs: Optional[Dict[str, Optional[str]]] = None
        self._option_text: List[str] = []
        self._textarea: Optional[str] = None
        self._textarea_text: List[str] = []
        self._submit_chosen = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if self._form_done:
            return
        attributes = dict(attrs)
        if tag == "form":
            if not self._form_done and self._form_depth == 0:
                self.action = attributes.get("action")
                self.enctype = (attributes.get("enctype") or self.enctype).strip().lower()
            self._form_depth += 1
            return
        if self._form_depth == 0:
            return
        if "disabled" in attributes:
            return
        name = attributes.get("name")
        if tag == "input" and name:
            kind = (attributes.get("type") or "").lower()
            declared = attributes.get("value")
            if kind in ("radio", "checkbox"):
                value = "on" if declared is None else declared
                self.offered_values.setdefault(name, []).append(value)
                if "checked" in attributes:
                    self.fields.append((name, value))
            elif kind in ("submit", "image"):
                self._submit_control(name, declared)
            elif kind not in ("reset", "button"):
                self.fields.append((name, "" if declared is None else declared))
        elif tag == "button" and name:
            if (attributes.get("type") or "").lower() not in ("button", "reset"):
                self._submit_control(name, attributes.get("value"))
        elif tag == "select" and name:
            self._select = name
            self._select_first = None
            self._select_chosen = None
            self.offered_values[name] = []
        elif tag == "option" and self._select is not None:
            self._finish_option()
            self._option_attrs = attributes
            self._option_text = []
        elif tag == "textarea" and name:
            self._textarea = name
            self._textarea_text = []

    def handle_data(self, data: str) -> None:
        if self._option_attrs is not None:
            self._option_text.append(data)
        if self._textarea is not None:
            self._textarea_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "option":
            self._finish_option()
        elif tag == "select" and self._select is not None:
            self._finish_option()
            chosen = self._select_chosen if self._select_chosen is not None else self._select_first
            if chosen is not None:
                self.fields.append((self._select, chosen))
            self._select = None
        elif tag == "textarea" and self._textarea is not None:
            self.fields.append((self._textarea, "".join(self._textarea_text)))
            self._textarea = None
        elif tag == "form" and self._form_depth > 0:
            self._form_depth -= 1
            if self._form_depth == 0:
                self._form_done = True

    def _submit_control(self, name: str, declared: Optional[str]) -> None:
        # MechanicalSoup's submit_selected() sends only the chosen submit
        # control -- by default the first -- and drops every other one; the
        # TRILEGAL form's second submit control is ``reset_form``.
        if not self._submit_chosen:
            self._submit_chosen = True
            self.fields.append((name, "" if declared is None else declared))

    def _finish_option(self) -> None:
        if self._option_attrs is None or self._select is None:
            self._option_attrs = None
            return
        if "disabled" in self._option_attrs:
            self._option_attrs = None
            return
        value = self._option_attrs.get("value")
        if value is None:
            value = "".join(self._option_text).strip()
        self.offered_values[self._select].append(value)
        if self._select_first is None:
            self._select_first = value
        if "selected" in self._option_attrs:
            self._select_chosen = value
        self._option_attrs = None


class LinkParser(HTMLParser):
    """Collect every ``<a href>`` of a page in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        href = dict(attrs).get("href") if tag == "a" else None
        if href:
            self.hrefs.append(href)


def submitted_form(page: str, overrides: Mapping[str, str]) -> Tuple[Optional[str], List[Tuple[str, str]], str]:
    """Return ``(action, fields, enctype)`` the first form of ``page`` submits with ``overrides``.

    Every override must name a field the form exposes, and a radio/checkbox/
    select override must be one of its offered values; an unclosed ``select``
    or ``textarea`` (a truncated page) is refused rather than dropping its
    value. All refusals raise ``TrilegalUnavailableError``.
    """
    parser = _TrilegalFormParser()
    parser.feed(page)
    parser.close()
    if parser._form_depth or parser._select is not None or parser._textarea is not None:
        raise TrilegalUnavailableError("TRILEGAL form page ends inside a form, select or textarea control")
    if not parser.fields and not parser.offered_values:
        raise TrilegalUnavailableError("TRILEGAL form page exposes no form controls")
    if parser.enctype not in ("application/x-www-form-urlencoded", "multipart/form-data"):
        raise TrilegalUnavailableError("TRILEGAL form encoding is unsupported: {0}".format(parser.enctype))
    names = {name for name, _value in parser.fields}
    names.update(parser.offered_values)
    missing = sorted(set(overrides) - names)
    if missing:
        raise TrilegalUnavailableError(
            "TRILEGAL form no longer exposes field(s): {0}".format(", ".join(missing))
        )
    for name, value in overrides.items():
        offered = parser.offered_values.get(name)
        if offered is not None and value not in offered:
            raise TrilegalUnavailableError(
                "TRILEGAL form field {0} does not offer {1!r}".format(name, value)
            )
    submitted: List[Tuple[str, str]] = []
    placed = set()
    for name, value in parser.fields:
        if name in overrides:
            if name not in placed:
                submitted.append((name, overrides[name]))
                placed.add(name)
            continue
        submitted.append((name, value))
    for name in overrides:
        if name not in placed:
            submitted.append((name, overrides[name]))
    return parser.action, submitted, parser.enctype

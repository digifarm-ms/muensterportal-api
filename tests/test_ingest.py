from muenster4you.ingest import clean_wikitext

TABLE_PAGE = """== Übersicht ==
{{Infobox|foo=bar}}
{| class="wikitable"
! Bezeichnung
! Adresse
! Kategorie
|-
| [[Sharing/GiveBoxen/Christuskirche|Christuskirche]]
| Ecke Hammerstraße/Friedrich-Ebert-Straße
| Givebox
|-
| Gievenbeck
| Bernings Kotten 9
| Givebox
|}
Text danach.<ref>Quelle</ref>
"""


def test_tables_become_one_line_per_row():
    text = clean_wikitext(TABLE_PAGE)

    assert "Bezeichnung | Adresse | Kategorie" in text
    assert "Christuskirche | Ecke Hammerstraße/Friedrich-Ebert-Straße | Givebox" in text
    assert "Gievenbeck | Bernings Kotten 9 | Givebox" in text


def test_templates_refs_and_markup_are_removed():
    text = clean_wikitext(TABLE_PAGE)

    assert "Infobox" not in text
    assert "Quelle" not in text
    assert "[[" not in text and "{|" not in text
    assert text.startswith("Übersicht")
    assert text.endswith("Text danach.")


def test_empty_and_plain_text_pass_through():
    assert clean_wikitext("") == ""
    assert clean_wikitext("Nur Text.") == "Nur Text."

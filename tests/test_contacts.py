"""Tests for contacts skill: contacts.find, permissions warning, case-insensitive and first-name match."""

from __future__ import annotations

from pathlib import Path

from ag.skills.builtin.contacts import find_contact


def test_contacts_find_unique_full_name(tmp_path: Path) -> None:
    contacts_file = tmp_path / "contacts.toml"
    contacts_file.write_text(
        """
[[contact]]
name = "Alice Smith"
email = "alice@example.com"

[[contact]]
name = "Bob Jones"
email = "bob@example.com"
""",
        encoding="utf-8",
    )

    res = find_contact("Alice Smith", workspace_dir=tmp_path)
    assert res["found"] is True
    assert res["count"] == 1
    assert res["match"]["email"] == "alice@example.com"
    assert res["match"]["name"] == "Alice Smith"


def test_contacts_find_first_name_and_case_insensitive(tmp_path: Path) -> None:
    contacts_file = tmp_path / "contacts.toml"
    contacts_file.write_text(
        """
[[contact]]
name = "Alice Smith"
email = "alice@example.com"
""",
        encoding="utf-8",
    )

    res = find_contact("alice", workspace_dir=tmp_path)
    assert res["found"] is True
    assert res["count"] == 1
    assert res["match"]["email"] == "alice@example.com"


def test_contacts_find_ambiguous_several_matches(tmp_path: Path) -> None:
    contacts_file = tmp_path / "contacts.toml"
    contacts_file.write_text(
        """
[[contact]]
name = "Alice Wonder"
email = "alice.w@example.com"

[[contact]]
name = "Alice Cooper"
email = "alice.c@example.com"
""",
        encoding="utf-8",
    )

    res = find_contact("Alice", workspace_dir=tmp_path)
    assert res["found"] is True
    assert res["count"] == 2
    assert res["ambiguous"] is True
    names = [c["name"] for c in res["candidates"]]
    assert "Alice Wonder" in names
    assert "Alice Cooper" in names


def test_contacts_find_max_5_cap(tmp_path: Path) -> None:
    contacts_file = tmp_path / "contacts.toml"
    content = "\n\n".join(
        f'[[contact]]\nname = "John {i}"\nemail = "john{i}@example.com"' for i in range(10)
    )
    contacts_file.write_text(content, encoding="utf-8")

    res = find_contact("John", workspace_dir=tmp_path)
    assert len(res["candidates"]) == 5


def test_contacts_find_none_found(tmp_path: Path) -> None:
    contacts_file = tmp_path / "contacts.toml"
    contacts_file.write_text(
        """
[[contact]]
name = "Alice Smith"
email = "alice@example.com"
""",
        encoding="utf-8",
    )

    res = find_contact("Charlie", workspace_dir=tmp_path)
    assert res["found"] is False
    assert res["count"] == 0


def test_contacts_file_permissions_warning(tmp_path: Path) -> None:
    contacts_file = tmp_path / "contacts.toml"
    contacts_file.write_text(
        '[[contact]]\nname = "Alice"\nemail = "alice@example.com"',
        encoding="utf-8",
    )
    # Make group/world writable
    contacts_file.chmod(0o666)

    res = find_contact("Alice", workspace_dir=tmp_path)
    assert res.get("warning") is not None
    assert "writable" in res["warning"].lower()

"""Tests for the TMDB poster lookup. No network: the HTTP call is faked."""

from src.posters import IMAGE_BASE, clean_title, find_poster, pick_poster


def test_clean_title_moves_article_and_drops_alias():
    assert clean_title("Close Shave, A (1995)") == ("A Close Shave", "1995")
    assert clean_title("Postino, Il (Postman, The) (1994)") == ("Il Postino", "1994")
    assert clean_title("Toy Story (1995)") == ("Toy Story", "1995")


def test_pick_poster_matches_year_within_one_and_skips_missing_posters():
    results = [
        {"poster_path": None, "release_date": "1995-01-01"},
        {"poster_path": "/wrong.jpg", "release_date": "2010-05-05"},
        {"poster_path": "/right.jpg", "release_date": "1996-02-02"},
    ]
    assert pick_poster(results, "1995") == IMAGE_BASE + "/right.jpg"
    assert pick_poster(results, "1950") is None


def test_find_poster_sends_the_cleaned_name_with_the_token():
    seen = {}

    def fake_get(url, token):
        seen["url"], seen["token"] = url, token
        return {"results": [{"poster_path": "/p.jpg", "release_date": "1995-03-03"}]}

    assert find_poster("A Close Shave", "1995", "secret", get=fake_get) == IMAGE_BASE + "/p.jpg"
    assert "A+Close+Shave" in seen["url"] and seen["token"] == "secret"

from reciperec.text import ingredient_analyzer, ingredient_tokens, normalize_ingredient, parse_list, split_sentences


def test_normalize_strips_quantities_units_and_prep():
    assert normalize_ingredient("1 1/2 cups soy sauce (low sodium)") == "soy sauce"
    assert normalize_ingredient("1 pound boneless, skinless chicken thighs") == "boneless skinless chicken thighs"
    assert normalize_ingredient("1 onion, finely chopped") == "onion"
    assert normalize_ingredient("½ teaspoon garlic powder") == "garlic powder"


def test_ingredient_tokens_join_phrases():
    assert ingredient_tokens(["2 tbsp soy sauce", "1 egg"]) == "soy_sauce egg"


def test_analyzer_pairs_stay_inside_one_ingredient():
    tokens = ingredient_analyzer("soy_sauce chicken_thighs")
    assert "soy sauce" in tokens and "chicken thighs" in tokens
    assert "sauce chicken" not in tokens


def test_parse_list_handles_json_python_and_plain_text():
    assert parse_list('["a", "b"]') == ["a", "b"]
    assert parse_list("['a', 'b']") == ["a", "b"]
    assert parse_list("a, b") == ["a", "b"]
    assert parse_list(float("nan")) == []


def test_split_sentences():
    assert split_sentences(["Preheat oven. Mix well!", "Bake 20 minutes."]) == [
        "Preheat oven.", "Mix well!", "Bake 20 minutes."]

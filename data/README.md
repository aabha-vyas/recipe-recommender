# Data

Put the recipe CSV here as `data/recipes.csv`. It isn't committed because it's about 80 MB.

Expected columns: `recipe_title`, `category`, `subcategory`, `description`,
`ingredients`, `directions`, `num_ingredients`, `num_steps`.
`ingredients` and `directions` are JSON-style lists of strings.

The raw file has 62,126 rows, but the same recipe often appears under several
categories. `reciperec.data.load_recipes` collapses those rows into one recipe
with all of its category and subcategory labels.

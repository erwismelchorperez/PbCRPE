import numpy as np

from srcc.pbcre.patterns.route_normalizer import (
    normalize_path,
    path_matches,
    RouteContradictionError
)


def test_numeric_interval():

    original = [
        (0, ">", 1374.0),
        (0, ">", 1000.0),
        (0, "<=", 2879.66),
        (1, "<=", 30.0),
        (1, "<=", 9.0)
    ]

    normalized = normalize_path(original)

    expected = [
        (0, ">", 1374.0),
        (0, "<=", 2879.66),
        (1, "<=", 9.0)
    ]

    assert normalized == expected


def test_categorical_redundancy():

    original = [
        (0, "!=", "A14"),
        (0, "=", "A12")
    ]

    normalized = normalize_path(original)

    expected = [
        (0, "=", "A12")
    ]

    assert normalized == expected


def test_grouped_categories():

    original = [
        (0, "in", ("A30", "A31", "A33")),
        (0, "not in", ("A31",))
    ]

    normalized = normalize_path(original)

    expected = [
        (0, "in", ("A30", "A33"))
    ]

    assert normalized == expected


def test_categorical_contradiction():

    original = [
        (0, "=", "A12"),
        (0, "!=", "A12")
    ]

    try:
        normalize_path(original)
        assert False, "Debió detectarse una contradicción."

    except RouteContradictionError:
        pass


def test_numeric_contradiction():

    original = [
        (0, ">", 100.0),
        (0, "<=", 100.0)
    ]

    try:
        normalize_path(original)
        assert False, "Debió detectarse una contradicción."

    except RouteContradictionError:
        pass


def test_missing_values():

    original = [
        (0, ">", 10.0),
        (1, "!=", "A14")
    ]

    normalized = normalize_path(original)

    # Ambas rutas deben aceptar los valores faltantes
    # en las dos características.
    instance = [np.nan, np.nan]

    assert path_matches(instance, original)
    assert path_matches(instance, normalized)


def test_equivalence():

    original = [
        (0, ">", 1374.0),
        (0, "<=", 2879.66),
        (1, "!=", "A14"),
        (1, "!=", "A12")
    ]

    normalized = normalize_path(original)

    instances = [
        [1500.0, "A10"],
        [2000.0, "A14"],
        [3000.0, "A10"],
        [np.nan, np.nan],
        [1500.0, np.nan],
        [np.nan, "A14"]
    ]

    for instance in instances:

        original_result = path_matches(
            instance,
            original
        )

        normalized_result = path_matches(
            instance,
            normalized
        )

        assert original_result == normalized_result


if __name__ == "__main__":

    test_numeric_interval()
    test_categorical_redundancy()
    test_grouped_categories()
    test_categorical_contradiction()
    test_numeric_contradiction()
    test_missing_values()
    test_equivalence()

    print("Todas las pruebas del normalizador fueron satisfactorias.")
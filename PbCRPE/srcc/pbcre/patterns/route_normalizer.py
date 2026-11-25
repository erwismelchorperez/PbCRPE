from collections import OrderedDict
from numbers import Number
import math


class RouteContradictionError(ValueError):
    """
    Se produce cuando las condiciones de una ruta
    son incompatibles entre sí.
    """
    pass


def _is_missing(value):
    """
    Comprueba si un valor es None o NaN.
    """
    if value is None:
        return True

    try:
        result = value != value

        if isinstance(result, bool):
            return result

        return bool(result)

    except (TypeError, ValueError):
        return False


def _same_value(value_a, value_b):
    """
    Comparación segura de valores categóricos.
    """
    if _is_missing(value_a) or _is_missing(value_b):
        return _is_missing(value_a) and _is_missing(value_b)

    try:
        result = value_a == value_b
        return bool(result)
    except (TypeError, ValueError):
        return False


def _contains(values, value):
    """
    Comprueba si un valor está contenido en una colección,
    utilizando comparación segura.
    """
    return any(_same_value(value, candidate) for candidate in values)


def _unique_values(values):
    """
    Elimina valores duplicados conservando su orden.
    """
    result = []

    for value in values:
        if not _contains(result, value):
            result.append(value)

    return result


def _sorted_values(values):
    """
    Ordena valores categóricos para obtener
    una representación determinista.
    """
    return tuple(sorted(values, key=lambda value: str(value)))


def _normalize_numeric_conditions(feature_idx, conditions):
    """
    Normaliza condiciones numéricas de una variable.

    Semántica original:

        x <= a  : rechaza valores faltantes.
        x > b   : acepta valores faltantes.

    Combina las condiciones en un intervalo cuando es posible.
    """

    lower_bounds = []
    upper_bounds = []

    for operator, threshold in conditions:

        if not isinstance(threshold, Number):
            raise TypeError(
                f"Umbral numérico inválido para la característica "
                f"{feature_idx}: {threshold!r}"
            )

        if operator == ">":
            lower_bounds.append(threshold)

        elif operator == "<=":
            upper_bounds.append(threshold)

        else:
            raise ValueError(
                f"Operador numérico no reconocido: {operator}"
            )

    normalized = []

    lower = max(lower_bounds) if lower_bounds else None
    upper = min(upper_bounds) if upper_bounds else None

    # Una intersección vacía representa una contradicción.
    if lower is not None and upper is not None:

        if lower >= upper:
            raise RouteContradictionError(
                f"Ruta contradictoria en característica {feature_idx}: "
                f"x > {lower} y x <= {upper}"
            )

    # Primero se representa el límite inferior.
    if lower is not None:
        normalized.append(
            (feature_idx, ">", lower)
        )

    # Después el límite superior.
    if upper is not None:
        normalized.append(
            (feature_idx, "<=", upper)
        )

    return normalized


def _normalize_categorical_conditions(feature_idx, conditions):
    """
    Normaliza condiciones categóricas.

    Operadores positivos:
        =, in

    Operadores negativos:
        !=, not in

    Conserva la semántica de valores faltantes:

        = y in       -> rechazan faltantes.
        != y not in  -> aceptan faltantes.
    """

    positive_constraints = []
    negative_constraints = []

    for operator, threshold in conditions:

        if operator == "=":
            positive_constraints.append([threshold])

        elif operator == "in":

            if not isinstance(
                threshold,
                (tuple, list, set, frozenset)
            ):
                raise TypeError(
                    f"Se esperaba una colección para 'in': "
                    f"{threshold!r}"
                )

            positive_constraints.append(list(threshold))

        elif operator == "!=":
            negative_constraints.append([threshold])

        elif operator == "not in":

            if not isinstance(
                threshold,
                (tuple, list, set, frozenset)
            ):
                raise TypeError(
                    f"Se esperaba una colección para 'not in': "
                    f"{threshold!r}"
                )

            negative_constraints.append(list(threshold))

        else:
            raise ValueError(
                f"Operador categórico no reconocido: {operator}"
            )

    # Unificar exclusiones.
    excluded_values = _unique_values(
        [
            value
            for constraint in negative_constraints
            for value in constraint
        ]
    )

    # Caso 1: existen restricciones positivas.
    if positive_constraints:

        allowed_values = list(positive_constraints[0])

        # Intersección de restricciones positivas.
        for constraint in positive_constraints[1:]:

            allowed_values = [
                value
                for value in allowed_values
                if _contains(constraint, value)
            ]

        # Aplicar exclusiones.
        allowed_values = [
            value
            for value in allowed_values
            if not _contains(excluded_values, value)
        ]

        if not allowed_values:
            raise RouteContradictionError(
                f"Ruta categórica contradictoria "
                f"en característica {feature_idx}."
            )

        allowed_values = _sorted_values(allowed_values)

        # Si queda un solo valor, utilizar igualdad.
        if len(allowed_values) == 1:

            return [
                (feature_idx, "=", allowed_values[0])
            ]

        return [
            (feature_idx, "in", allowed_values)
        ]

    # Caso 2: solamente existen restricciones negativas.
    # Se conserva la aceptación de valores faltantes.
    if excluded_values:

        excluded_values = _sorted_values(excluded_values)

        if len(excluded_values) == 1:

            return [
                (feature_idx, "!=", excluded_values[0])
            ]

        return [
            (feature_idx, "not in", excluded_values)
        ]

    return []

def normalize_path(path):
    """
    Normaliza una ruta del árbol PBCRETree.

    Parámetros
    ----------
    path : iterable
        Colección de condiciones representadas como:

        (feature_idx, operator, threshold)

    Operadores admitidos
    --------------------
    Numéricos:
        <=, >

    Categóricos:
        =, !=, in, not in

    Retorno
    -------
    list
        Ruta normalizada.

    Excepciones
    ----------
    RouteContradictionError
        Si las condiciones de una variable son incompatibles.
    """

    grouped_conditions = OrderedDict()

    # Agrupamos condiciones por característica.
    for condition in path:

        if len(condition) != 3:
            raise ValueError(
                f"Condición inválida: {condition!r}"
            )

        feature_idx, operator, threshold = condition

        if feature_idx not in grouped_conditions:
            grouped_conditions[feature_idx] = []

        grouped_conditions[feature_idx].append(
            (operator, threshold)
        )

    normalized_path = []

    numeric_operators = {"<=", ">"}
    categorical_operators = {"=", "!=", "in", "not in"}

    for feature_idx, conditions in grouped_conditions.items():

        operators = {
            operator
            for operator, _ in conditions
        }

        if operators.issubset(numeric_operators):

            normalized_conditions = _normalize_numeric_conditions(
                feature_idx,
                conditions
            )

        elif operators.issubset(categorical_operators):

            normalized_conditions = _normalize_categorical_conditions(
                feature_idx,
                conditions
            )

        else:
            raise ValueError(
                f"Combinación de operadores incompatible "
                f"en característica {feature_idx}: {operators}"
            )

        normalized_path.extend(normalized_conditions)

    return normalized_path


def path_matches(instance, path):
    """
    Comprueba si una instancia satisface todas las condiciones
    de una ruta.

    Esta función reproduce la semántica de _match_path()
    de PBCRETree para los operadores utilizados por el árbol.

    Parámetros
    ----------
    instance : array-like o Series
        Instancia que se desea evaluar.

    path : iterable
        Ruta original o normalizada.

    Retorno
    -------
    bool
        True si la instancia satisface todas las condiciones.
    """

    for feature_idx, operator, threshold in path:

        value = instance[feature_idx]

        # Condiciones numéricas.
        if operator == "<=":

            if _is_missing(value):
                return False

            if float(value) > float(threshold):
                return False

        elif operator == ">":

            # El árbol acepta valores faltantes por esta rama.
            if _is_missing(value):
                continue

            if float(value) <= float(threshold):
                return False

        # Condiciones categóricas individuales.
        elif operator == "=":

            if _is_missing(value):
                return False

            if not _same_value(value, threshold):
                return False

        elif operator == "!=":

            # El árbol acepta valores faltantes por esta rama.
            if _is_missing(value):
                continue

            if _same_value(value, threshold):
                return False

        # Condiciones categóricas agrupadas.
        elif operator == "in":

            if _is_missing(value):
                return False

            if not _contains(threshold, value):
                return False

        elif operator == "not in":

            # El árbol acepta valores faltantes por esta rama.
            if _is_missing(value):
                continue

            if _contains(threshold, value):
                return False

        else:
            raise ValueError(
                f"Operador no reconocido: {operator}"
            )

    return True
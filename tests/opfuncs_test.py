import math
import re
from typing import Callable

import numpy as np
import pytest

from opfuncs import SUPPORTED_OPERATORS
from opfuncs import opfunc

Vector3D = tuple[float, float, float]


# --------
# fixtures
# --------


@pytest.fixture
def add_5():
    def _add_5(x):
        return x + 5

    return _add_5


@pytest.fixture
def times_3():
    def _times_3(x):
        return 3 * x

    return _times_3


@pytest.fixture
def polynomial():
    def _polynomial(x):
        return x**2 + 2 * x - 3

    return _polynomial


@pytest.fixture
def floordiv_by_4():
    def _floordiv_by_4(x):
        return x // 4

    return _floordiv_by_4


@pytest.fixture
def np_transpose():
    def _np_transpose(x: np.ndarray) -> np.ndarray:
        return x.T

    return _np_transpose


@pytest.fixture
def np_add_matrix():
    def _np_add_matrix(x: np.ndarray) -> np.ndarray:
        return x + np.array([[1, 2], [3, 4]])

    return _np_add_matrix


@pytest.fixture
def np_fill_diag():
    def _np_fill_diag(x: np.ndarray) -> np.ndarray:
        np.fill_diagonal(x, 1)
        return x

    return _np_fill_diag


@pytest.fixture
def only_evens():
    def _only_evens(s: set[int]) -> set[int]:
        return {i for i in s if i % 2 == 0}

    return _only_evens


@pytest.fixture
def perfect_squares():
    def _perfect_squares(s: set[int]) -> set[int]:
        return {i for i in s if math.sqrt(i) == int(math.sqrt(i)) and i > 0}

    return _perfect_squares


@pytest.fixture
def less_than_7():
    def _less_than_7(s: set[int]) -> set[int]:
        return {i for i in s if i < 7}

    return _less_than_7


@pytest.fixture
def double_vec():
    def _double_vec(a: Vector3D) -> Vector3D:
        return 2 * a[0], 2 * a[1], 2 * a[2]

    return _double_vec


@pytest.fixture
def invert_z():
    def _invert_z(a: Vector3D) -> Vector3D:
        return a[0], a[1], -a[2]

    return _invert_z


@pytest.fixture
def swap_x_and_y():
    def _swap_x_and_y(a: Vector3D) -> Vector3D:
        return a[1], a[0], -a[2]

    return _swap_x_and_y


@pytest.fixture
def vector_custom_ops():
    def dot_product_op_builder(
        func1: Callable[[Vector3D], Vector3D],
        func2: Callable[[Vector3D], Vector3D],
    ) -> Callable[[Vector3D], float]:
        def dot_product_op(a: Vector3D) -> float:
            a1 = func1(a)
            a2 = func2(a)

            return math.fsum(a1[i] * a2[i] for i in range(3))

        return dot_product_op

    def cross_product_op_builder(
        func1: Callable[[Vector3D], Vector3D],
        func2: Callable[[Vector3D], Vector3D],
    ) -> Callable[[Vector3D], Vector3D]:
        def cross_product_op(a: Vector3D) -> Vector3D:
            a1 = func1(a)
            a2 = func2(a)

            x = (a1[1] * a2[2]) - (a2[1] * a1[2])
            y = (a1[2] * a2[0]) - (a2[2] * a1[0])
            z = (a1[0] * a2[1]) - (a2[0] * a1[1])

            return x, y, z

        return cross_product_op

    def round_op_builder(
        func: Callable[[Vector3D], Vector3D],
        n: int | None = None,
    ) -> Callable[[Vector3D], Vector3D]:
        def round_op(a: Vector3D) -> Vector3D:
            b = func(a)

            x = round(b[0], n)
            y = round(b[1], n)
            z = round(b[2], n)

            return x, y, z

        return round_op

    return {
        "mul": dot_product_op_builder,
        "matmul": cross_product_op_builder,
        "round": round_op_builder,
    }


# ----------
# test cases
# ----------


def test_supported_operators():
    assert set(SUPPORTED_OPERATORS) == {
        "and",
        "or",
        "xor",
        "add",
        "sub",
        "mul",
        "matmul",
        "truediv",
        "floordiv",
        "mod",
        "divmod",
        "pow",
        "lshift",
        "rshift",
        "neg",
        "pos",
        "abs",
        "invert",
        "round",
        "trunc",
        "floor",
        "ceil",
    }


def test_wrapping(add_5, times_3, polynomial):
    # the name of the inner functions of the pytest fixtures
    # are returned by __name__ as opposed to the name of the
    # pytest fixture, due to pytest filling the fixture parameters
    # in the test case with the return values of the fixtures.

    assert opfunc(add_5)
    assert opfunc(add_5).__name__ == add_5.__name__ == "_add_5"
    assert opfunc(add_5)(4) == add_5(4) == 9

    assert opfunc()(times_3)
    assert opfunc()(times_3).__name__ == times_3.__name__ == "_times_3"
    assert opfunc()(times_3)(2.5) == times_3(2.5) == 7.5

    assert opfunc(opfunc(polynomial))
    assert opfunc(opfunc(polynomial)).__name__ == polynomial.__name__ == "_polynomial"
    assert opfunc(opfunc(polynomial))(-3) == polynomial(-3) == 0


def test_decorator():
    @opfunc
    def prefix_hello(s: str) -> str:
        return "Hello " + s

    assert prefix_hello.__name__ == "prefix_hello"
    assert prefix_hello("World") == "Hello World"

    @opfunc()
    def remove_e(s: str) -> str:
        return s.replace("e", "")

    assert remove_e.__name__ == "remove_e"
    assert remove_e("Welcome!") == "Wlcom!"


def test_and_operator(only_evens, less_than_7, perfect_squares):
    opfunc_only_evens = opfunc(only_evens)
    opfunc_less_than_7 = opfunc(less_than_7)
    opfunc_perfect_squares = opfunc(perfect_squares)

    s = set(i for i in range(1, 21))

    assert (func1 := opfunc_only_evens & opfunc_less_than_7)
    assert func1(s) == {2, 4, 6}

    assert (func2 := opfunc_less_than_7 & perfect_squares)
    assert func2(s) == {1, 4}

    error_text = "unsupported operand type(s) for &: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        only_evens & opfunc_perfect_squares


def test_or_operator(only_evens, less_than_7, perfect_squares):
    opfunc_only_evens = opfunc(only_evens)
    opfunc_less_than_7 = opfunc(less_than_7)
    opfunc_perfect_squares = opfunc(perfect_squares)

    s = set(i for i in range(1, 21))

    assert (func1 := opfunc_less_than_7 | opfunc_perfect_squares)
    assert func1(s) == {1, 2, 3, 4, 5, 6, 9, 16}

    assert (func2 := opfunc_perfect_squares | only_evens)
    assert func2(s) == {1, 2, 4, 6, 8, 9, 10, 12, 14, 16, 18, 20}

    error_text = "unsupported operand type(s) for |: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        less_than_7 | opfunc_only_evens


def test_xor_operator(only_evens, less_than_7, perfect_squares):
    opfunc_only_evens = opfunc(only_evens)
    opfunc_less_than_7 = opfunc(less_than_7)
    opfunc_perfect_squares = opfunc(perfect_squares)

    s = set(i for i in range(1, 21))

    assert (func1 := opfunc_perfect_squares ^ opfunc_only_evens)
    assert func1(s) == {1, 2, 6, 8, 9, 10, 12, 14, 18, 20}

    assert (func2 := opfunc_only_evens ^ less_than_7)
    assert func2(s) == {1, 3, 5, 8, 10, 12, 14, 16, 18, 20}

    error_text = "unsupported operand type(s) for ^: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        perfect_squares ^ opfunc_less_than_7


def test_add_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := opfunc_add_5 + opfunc_times_3)
    assert func1(7) == 33

    assert (func2 := opfunc_times_3 + polynomial)
    assert func2(3) == 21

    error_text = "unsupported operand type(s) for +: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        add_5 + opfunc_polynomial


def test_sub_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := opfunc_times_3 - opfunc_polynomial)
    assert func1(7) == -39

    assert (func2 := opfunc_polynomial - add_5)
    assert func2(3) == 4

    error_text = "unsupported operand type(s) for -: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        times_3 - opfunc_add_5


def test_mul_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := opfunc_polynomial * opfunc_add_5)
    assert func1(7) == 720

    assert (func2 := opfunc_add_5 * times_3)
    assert func2(3) == 72

    error_text = "unsupported operand type(s) for *: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        polynomial * opfunc_times_3


# afaik, no object in the python stdlib implements __matmul__,
# so we use numpy as a test-dependency to test matmul.
def test_matmul_operator(np_transpose, np_add_matrix, np_fill_diag):
    opfunc_np_transpose = opfunc(np_transpose)
    opfunc_np_add_matrix = opfunc(np_add_matrix)
    opfunc_np_fill_diag = opfunc(np_fill_diag)

    matrix = np.array([[3, 1], [-1, 2]])

    assert (func1 := opfunc_np_transpose @ opfunc_np_add_matrix)
    assert np.array_equal(func1(matrix), np.array([[10, 3], [8, 15]]))

    assert (func2 := opfunc_np_add_matrix @ np_fill_diag)
    assert np.array_equal(func2(matrix), np.array([[1, 7], [-4, 8]]))

    error_text = "unsupported operand type(s) for @: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        np_transpose @ opfunc_np_fill_diag


def test_truediv_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := opfunc_add_5 / opfunc_times_3)
    assert func1(7) == 4 / 7

    assert (func2 := opfunc_times_3 / polynomial)
    assert func2(3) == 0.75

    error_text = "unsupported operand type(s) for /: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        add_5 / opfunc_polynomial


def test_floordiv_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := opfunc_polynomial // opfunc_add_5)
    assert func1(7) == 5

    assert (func2 := opfunc_polynomial // times_3)
    assert func2(3) == 1

    error_text = "unsupported operand type(s) for //: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        add_5 // opfunc_times_3


def test_mod_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := opfunc_polynomial % opfunc_times_3)
    assert func1(7) == 18

    assert (func2 := opfunc_times_3 % add_5)
    assert func2(3) == 1

    error_text = "unsupported operand type(s) for %: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        polynomial % opfunc_add_5


def test_divmod_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := divmod(opfunc_polynomial, opfunc_times_3))
    assert func1(7) == (2, 18)

    assert (func2 := divmod(opfunc_times_3, add_5))
    assert func2(3) == (1, 1)

    error_text = "unsupported operand type(s) for divmod(): " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        divmod(polynomial, opfunc_add_5)


def test_pow_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := opfunc_polynomial**opfunc_add_5)
    assert func1(7) == 2176782336000000000000

    assert (func2 := opfunc_times_3**add_5)
    assert func2(3) == 43046721

    assert (func3 := pow(opfunc_add_5, opfunc_polynomial))
    assert func3(-2) == 1 / 27

    assert (func3 := pow(opfunc_times_3, add_5))
    assert func3(-3) == 81

    assert (func3 := pow(opfunc_times_3, opfunc_add_5, 7))
    assert func3(-2) == 1

    assert (func3 := pow(opfunc_times_3, add_5, 7))
    assert func3(-3) == 4

    error_text = (
        "unsupported operand type(s) for ** or pow(): " "'function' and '_OpFunc'"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        times_3**opfunc_polynomial

    error_text = (
        "unsupported operand type(s) for ** or pow(): " "'function' and '_OpFunc'"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        pow(times_3, opfunc_polynomial)

    error_text = (
        "unsupported operand type(s) for ** or pow(): " "'function', '_OpFunc', 'int'"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        pow(times_3, opfunc_polynomial, 6)


def test_lshift_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := opfunc_polynomial << opfunc_add_5)
    assert func1(7) == 245760

    assert (func2 := opfunc_add_5 << times_3)
    assert func2(3) == 4096

    error_text = "unsupported operand type(s) for <<: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        polynomial << opfunc_times_3


def test_rshift_operator(floordiv_by_4, times_3, polynomial):
    opfunc_floordiv_by_4 = opfunc(floordiv_by_4)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := opfunc_polynomial >> opfunc_floordiv_by_4)
    assert func1(8) == 19

    assert (func2 := opfunc_times_3 >> floordiv_by_4)
    assert func2(4) == 6

    error_text = "unsupported operand type(s) for >>: " "'function' and '_OpFunc'"

    with pytest.raises(TypeError, match=re.escape(error_text)):
        times_3 >> opfunc_polynomial


def test_neg_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := -opfunc_add_5)
    assert func1(7) == -12

    assert (func2 := -opfunc_times_3)
    assert func2(3) == -9

    assert (func3 := -opfunc_polynomial)
    assert func3(4) == -21


def test_pos_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := +opfunc_add_5)
    assert func1(7) == 12

    assert (func2 := +opfunc_times_3)
    assert func2(3) == 9

    assert (func3 := +opfunc_polynomial)
    assert func3(4) == 21


def test_abs_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := abs(opfunc_add_5))
    assert func1(-16) == 11

    assert (func2 := abs(opfunc_times_3))
    assert func2(-3.5) == 10.5

    assert (func3 := abs(opfunc_polynomial))
    assert func3(-2) == 3


def test_invert_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := ~opfunc_add_5)
    assert func1(7) == -13

    assert (func2 := ~opfunc_times_3)
    assert func2(3) == -10

    assert (func3 := ~opfunc_polynomial)
    assert func3(4) == -22


def test_round_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := round(opfunc_add_5))
    assert func1(6.5) == 12

    assert (func2 := round(opfunc_times_3, 2))
    assert func2(-5.231) == -15.69

    assert (func3 := round(opfunc_polynomial, 1))
    assert func3(-2.5) == -1.8


def test_trunc_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := math.trunc(opfunc_add_5))
    assert func1(6.5) == 11

    assert (func2 := math.trunc(opfunc_times_3))
    assert func2(-5.2) == -15

    assert (func3 := math.trunc(opfunc_polynomial))
    assert func3(-2.5) == -1


def test_floor_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := math.floor(opfunc_add_5))
    assert func1(6.5) == 11

    assert (func2 := math.floor(opfunc_times_3))
    assert func2(-5.2) == -16

    assert (func3 := math.floor(opfunc_polynomial))
    assert func3(-2.5) == -2


def test_ceil_operator(add_5, times_3, polynomial):
    opfunc_add_5 = opfunc(add_5)
    opfunc_times_3 = opfunc(times_3)
    opfunc_polynomial = opfunc(polynomial)

    assert (func1 := math.ceil(opfunc_add_5))
    assert func1(6.5) == 12

    assert (func2 := math.ceil(opfunc_times_3))
    assert func2(-5.2) == -15

    assert (func3 := math.ceil(opfunc_polynomial))
    assert func3(-2.5) == -1


def test_include(only_evens, less_than_7, perfect_squares):
    opfunc_less_than_7_and = opfunc(less_than_7, include="and")
    opfunc_less_than_7_or = opfunc(less_than_7, include="or")
    opfunc_only_evens = opfunc(only_evens, include=["and", "or"])
    opfunc_perfect_squares = opfunc(perfect_squares, include="or")

    # checks that operators do not raise a TypeError for
    # valid operations between two opfuncs that specify
    # 'include'. If the second function is not an opfunc
    # or is an opfunc that did not specify 'include',
    # then only operators in the first opfunc's 'include'
    # list are considered.

    assert opfunc_less_than_7_and & opfunc_only_evens
    assert opfunc_only_evens | opfunc_perfect_squares
    assert opfunc_less_than_7_or | opfunc_perfect_squares
    assert opfunc_less_than_7_and & perfect_squares
    assert opfunc_less_than_7_and & opfunc(perfect_squares)

    # checks that TypeError is raised whenever an invalid
    # operator is used between two opfuncs that specify
    # 'include'. If the second function is not an opfunc
    # or is an opfunc that did not specify 'include',
    # then only operators in the first opfunc's 'include'
    # list are considered.

    error_text = (
        "unsupported operand type(s) for |: "
        "'_OpFunc' ('_only_evens') and '_OpFunc' ('_less_than_7')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_only_evens | opfunc_less_than_7_and

    error_text = (
        "unsupported operand type(s) for &: "
        "'_OpFunc' ('_less_than_7') and '_OpFunc' ('_perfect_squares')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_less_than_7_and & opfunc_perfect_squares

    error_text = (
        "unsupported operand type(s) for &: "
        "'_OpFunc' ('_perfect_squares') and '_OpFunc' ('_less_than_7')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_perfect_squares & opfunc_less_than_7_and

    error_text = (
        "unsupported operand type(s) for |: "
        "'_OpFunc' ('_perfect_squares') and '_OpFunc' ('_less_than_7')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_perfect_squares | opfunc_less_than_7_and

    error_text = (
        "unsupported operand type(s) for ^: "
        "'_OpFunc' ('_less_than_7') and 'function' ('_only_evens')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_less_than_7_and ^ only_evens

    error_text = (
        "unsupported operand type(s) for ^: "
        "'_OpFunc' ('_less_than_7') and '_OpFunc' ('_only_evens')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_less_than_7_and ^ opfunc(only_evens)

    # checks that composite opfuncs only includes
    # operations allowed by both sub-functions.

    func = opfunc_less_than_7_and & opfunc_only_evens
    with pytest.raises(TypeError):
        func | opfunc_perfect_squares

    func = opfunc_only_evens & opfunc_less_than_7_and
    with pytest.raises(TypeError):
        func | opfunc_perfect_squares


def test_exclude(less_than_7, only_evens, perfect_squares):
    opfunc_less_than_7_not_or = opfunc(less_than_7, exclude="or")
    opfunc_less_than_7_not_and = opfunc(less_than_7, exclude="and")
    opfunc_only_evens = opfunc(only_evens, exclude=["or", "xor"])
    opfunc_perfect_squares = opfunc(perfect_squares, exclude="or")

    # checks that operators do not raise a TypeError for
    # valid operations between two opfuncs that specify
    # 'exclude'. If the second function is not an opfunc
    # or is an opfunc that did not specify 'exclude',
    # then only operators in the first opfunc's 'exclude'
    # list are considered.

    assert opfunc_less_than_7_not_or & opfunc_only_evens
    assert opfunc_only_evens & opfunc_perfect_squares
    assert opfunc_less_than_7_not_or ^ opfunc_perfect_squares
    assert opfunc_less_than_7_not_and | perfect_squares
    assert opfunc_less_than_7_not_and | opfunc(perfect_squares)

    # checks that TypeError is raised whenever an invalid
    # operator is used between two opfuncs that specify
    # 'exclude'. If the second function is not an opfunc
    # or is an opfunc that did not specify 'exclude',
    # then only operators in the first opfunc's 'exclude'
    # list are considered.

    error_text = (
        "unsupported operand type(s) for |: "
        "'_OpFunc' ('_less_than_7') and 'function' ('_only_evens')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_less_than_7_not_or | only_evens

    error_text = (
        "unsupported operand type(s) for ^: "
        "'_OpFunc' ('_less_than_7') and '_OpFunc' ('_only_evens')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_less_than_7_not_or ^ opfunc_only_evens

    error_text = (
        "unsupported operand type(s) for |: "
        "'_OpFunc' ('_less_than_7') and '_OpFunc' ('_perfect_squares')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_less_than_7_not_or | opfunc_perfect_squares

    error_text = (
        "unsupported operand type(s) for |: "
        "'_OpFunc' ('_perfect_squares') and '_OpFunc' ('_less_than_7')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_perfect_squares | opfunc_less_than_7_not_and

    error_text = (
        "unsupported operand type(s) for |: "
        "'_OpFunc' ('_less_than_7') and '_OpFunc' ('_perfect_squares')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_less_than_7_not_and | opfunc_perfect_squares

    # checks that the composite function always excludes
    # operations excluded by either sub-function.

    func = opfunc_less_than_7_not_or ^ opfunc_perfect_squares
    with pytest.raises(TypeError):
        func ^ opfunc_only_evens


def test_custom_ops(double_vec, invert_z, swap_x_and_y, vector_custom_ops):
    opfunc_double_vec = opfunc(double_vec, custom_ops=vector_custom_ops)
    opfunc_invert_z = opfunc(invert_z, custom_ops=vector_custom_ops)
    opfunc_swap_x_and_y = opfunc(swap_x_and_y, custom_ops=vector_custom_ops)

    assert (volume := abs((opfunc_double_vec @ invert_z) * opfunc_swap_x_and_y))
    assert volume((1, 2, 3)) == 36

    assert (func := round(opfunc_invert_z, 3))
    assert func((2.1429548, -3.423477, 4.62)) == (2.143, -3.423, -4.62)

    error_text = (
        "unsupported operand type(s) for custom @: "
        "'_OpFunc' ('_swap_x_and_y') and '_OpFunc' ('_double_vec')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc_swap_x_and_y @ opfunc(double_vec)

    error_text = (
        "unsupported operand type(s) for custom @: "
        "'_OpFunc' ('_double_vec') and '_OpFunc' ('_swap_x_and_y')"
    )
    with pytest.raises(TypeError, match=re.escape(error_text)):
        opfunc(double_vec) @ opfunc_swap_x_and_y

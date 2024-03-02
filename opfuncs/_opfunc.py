from __future__ import annotations

__all__ = [
    'SUPPORTED_OPERATORS',
    'opfunc',
]

import functools
import math
import operator
from collections.abc import Callable
from collections.abc import Sequence
from typing import Concatenate
from typing import Generic
from typing import Literal
from typing import NamedTuple
from typing import ParamSpec
from typing import TypeVar
from typing import overload
from collections.abc import Mapping

FuncParams = ParamSpec('FuncParams')
OpParams = ParamSpec('OpParams')
T = TypeVar('T')

Func = Callable[FuncParams, T]

UnaryOpBuilder = Callable[Concatenate[Func, OpParams], Func]
BinaryOpBuilder = Callable[Concatenate[Func, Func, OpParams], Func]
CustomOpBuilder = UnaryOpBuilder | BinaryOpBuilder

UnaryOperator = Callable[Concatenate[T, OpParams], T]
BinaryOperator = Callable[Concatenate[T, T, OpParams], T]


class _OpData(NamedTuple):
    name: str
    op: UnaryOperator | BinaryOperator
    symbol: str


OPERATOR_DATA = [
    _OpData('and', operator.and_, '&'),
    _OpData('or', operator.or_, '|'),
    _OpData('xor', operator.xor, '^'),
    _OpData('add', operator.add, '+'),
    _OpData('sub', operator.sub, '-'),
    _OpData('mul', operator.mul, '*'),
    _OpData('matmul', operator.matmul, '@'),
    _OpData('truediv', operator.truediv, '/'),
    _OpData('floordiv', operator.floordiv, '//'),
    _OpData('mod', operator.mod, '%'),
    _OpData('divmod', divmod, 'divmod()'),
    _OpData('pow', pow, '** or pow()'),
    _OpData('lshift', operator.lshift, '<<'),
    _OpData('rshift', operator.rshift, '>>'),
    _OpData('neg', operator.neg, '-()'),
    _OpData('pos', operator.pos, '+()'),
    _OpData('abs', operator.abs, 'abs()'),
    _OpData('invert', operator.invert, '~'),
    _OpData('round', round, 'round()'),
    _OpData('trunc', math.trunc, 'math.trunc()'),
    _OpData('floor', math.floor, 'math.floor()'),
    _OpData('ceil', math.ceil, 'math.ceil()'),
]
OPERATORS_DICT = {data.name: data for data in OPERATOR_DATA}
SUPPORTED_OPERATORS = [data.name for data in OPERATOR_DATA]


@overload
def opfunc(
    func: None = ...,
    *,
    include: str | Sequence[str] | None = ...,
    exclude: str | Sequence[str] | None = ...,
    custom_ops: Mapping[str, CustomOpBuilder] | None = ...,
) -> Callable[[Func], _OpFunc[FuncParams, T]]: ...


@overload
def opfunc(
    func: Func,
    *,
    include: str | Sequence[str] | None = ...,
    exclude: str | Sequence[str] | None = ...,
    custom_ops: Mapping[str, CustomOpBuilder] | None = ...,
) -> _OpFunc[FuncParams, T]: ...


def opfunc(
    func: Func | None = None,
    *,
    include: str | Sequence[str] | None = None,
    exclude: str | Sequence[str] | None = None,
    custom_ops: Mapping[str, CustomOpBuilder] | None = None,
) -> _OpFunc[FuncParams, T] | Callable[[Func], _OpFunc[FuncParams, T]]:
    """Wraps or decorates a function, enabling operator usage on the
    function (e.g. ``+``, ``*``, etc.). Allowed operators can be
    restricted using the *include* and *exclude* parameters. The full
    list of supported operators can be found in
    ``opfuncs.SUPPORTED_OPERATORS``. Functions wrapped with opfunc can
    also be composed with other opfuncs or callables using square
    brackets ('``[]``'). The *custom_ops* parameter can be specified
    to use custom operators in place of the standard operators.

    Basic Usage
    -----------

    >>> @opfunc
    ... def a(x: float) -> float:
    ...     return x + 3
    ...
    >>> @opfunc
    ... def b(x: float) -> float:
    ...     return 2 * x
    ...

    >>> (a + b)(5)  # == a(5) + b(5)
    18

    >>> (a * b)(4)  # == a(4) * b(4)
    56

    Using *include* and *exclude*
    -----------------------------

    >>> @opfunc(include=("add", "sub"))
    ... def c(x: float) -> float:
    ...     return x / 4
    ...
    >>> @opfunc(exclude=("mul", "truediv"))
    >>> def d(x: float) -> float:
    ...     return x ** 2
    ...

    >>> (c + d)(2)  # == c(2) + d(2)
    4.5

    >>> (a * c)(3)  # raises TypeError
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for *: ...

    >>> (d / b)(4)  # raises TypeError
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for /: ...

    Using composition
    -----------------

    >>> a[b](7)  # == a(b(7))
    17
    >>> b[3](3)  # == b(b(b(3)))
    24

    Using alongside normal functions
    --------------------------------

    When using an opfunc alongside a normal function, the order
    matters. The normal function must be on the rightside of the
    operation.

    >>> def e(x: float) -> float:
    ...     return x - 10

    >>> (a * e)(5)  # == a(5) * e(5)
    -40

    >>> b[e](4)  # == b(e(4))
    -12

    >>> (e + c)(5)  # raises TypeError
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for +: ...

    Using custom operators
    ----------------------

    >>> def custom_xor(
    ...     f: Callable[[float], float],
    ...     g: Callable[[float], float],
    ... ) -> Callable[[float], float]:
    ...     def xor_as_pow(x: float) -> float:
    ...         return f(x) ** g(x)
    ...
    ...     return xor_as_pow
    ...

    >>> @opfunc(custom_ops={"xor": custom_xor})
    ... def f(x: float) -> float:
    ...     return x / 2
    ...

    >>> @opfunc(custom_ops={"xor": custom_xor})
    ... def g(x: float) -> float:
    ...     return x + 4
    ...

    >>> (g ^ f)(4)
    64.0

    :param func: (optional) Callable or ``None``. If ``None``, a
                 new wrapper function will be returned that
                 transforms a callable into an opfunc.

    :param include: (optional) Operator or list of operators to
                    include. If ``None``, all operators are included.

    :param exclude: (optional) Operator or list of operators to
                    exclude. If ``None``, no operators are excluded.

    :param custom_ops: (optional) A mapping of operator names to
                       functions that build custom operators. The
                       built custom operators will be used in place
                       of their respective standard operators.

    :return: Wrapped function that can use python's standard
             operators.
    """
    if func is None:

        def _opfunc_partial(__func: Func, /):
            return opfunc(
                __func,
                include=include,
                exclude=exclude,
                custom_ops=custom_ops,
            )

        return _opfunc_partial

    settings = _OpFuncSettings.new(
        include=include,
        exclude=exclude,
        custom_ops=custom_ops,
    )

    if isinstance(func, _OpFunc):
        func = func._func

    _opfunc: _OpFunc[FuncParams, T] = _OpFunc(func, settings)
    _opfunc = functools.wraps(func)(_opfunc)

    return _opfunc


class _OpFunc(Generic[FuncParams, T]):
    def __init__(
        self,
        func: Func,
        settings: _OpFuncSettings,
    ):
        self._func = func
        self._settings = settings

    def __call__(
        self,
        *args: FuncParams.args,
        **kwargs: FuncParams.kwargs,
    ) -> T:
        return self._func(*args, **kwargs)

    def __and__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('and', other)

    def __or__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('or', other)

    def __xor__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('xor', other)

    def __add__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('add', other)

    def __sub__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('sub', other)

    def __mul__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('mul', other)

    def __matmul__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('matmul', other)

    def __truediv__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('truediv', other)

    def __floordiv__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('floordiv', other)

    def __mod__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('mod', other)

    def __divmod__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('divmod', other)

    def __pow__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
        modulo: int | None = None,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('pow', other, modulo)

    def __lshift__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('lshift', other)

    def __rshift__(
        self,
        other: _OpFunc[FuncParams, T] | Func,
    ) -> _OpFunc[FuncParams, T]:
        return self._operate('rshift', other)

    def __neg__(self) -> _OpFunc[FuncParams, T]:
        return self._operate('neg')

    def __pos__(self) -> _OpFunc[FuncParams, T]:
        return self._operate('pos')

    def __abs__(self) -> _OpFunc[FuncParams, T]:
        return self._operate('abs')

    def __invert__(self) -> _OpFunc[FuncParams, T]:
        return self._operate('invert')

    def __round__(self, n: int | None = None) -> _OpFunc[FuncParams, T]:
        return self._operate('round', None, n)

    def __trunc__(self) -> _OpFunc[FuncParams, T]:
        return self._operate('trunc')

    def __floor__(self) -> _OpFunc[FuncParams, T]:
        return self._operate('floor')

    def __ceil__(self) -> _OpFunc[FuncParams, T]:
        return self._operate('ceil')

    def __getitem__(self, __func: int | Callable, /) -> _OpFunc[FuncParams, T]:
        if not isinstance(__func, int) and not callable(__func):
            raise TypeError(
                f"cannot compose opfunc {self._func.__name__!r} with "
                f"{__func.__class__.__name__!r} object {__func!r}.",
            )

        if not isinstance(__func, int):
            return self._compose_with(__func)

        if __func <= 0:
            raise ValueError(
                f"invalid integer input for opfunc composition of "
                f"{self._func.__name__!r}: {__func}. Expected a "
                f"positive integer.",
            )

        _opfunc = self
        for _ in range(__func - 1):
            _opfunc = self._compose_with(_opfunc)

        return _opfunc

    def _compose_with(self, func: Callable) -> _OpFunc:
        def composite_func(
            *args: FuncParams.args,
            **kwargs: FuncParams.kwargs,
        ):
            return self(func(*args, **kwargs))

        return _OpFunc(
            composite_func,
            self._settings,
        )

    def _operate(
        self,
        op_name: str,
        other: _OpFunc[FuncParams, T] | Func | None = None,
        *op_args: OpParams.args,
        **op_kwargs: OpParams.kwargs,
    ) -> _OpFunc[FuncParams, T]:
        if isinstance(other, _OpFunc):
            merged_settings = self._settings.merge_settings(other._settings)
            other_custom_ops = other._settings.custom_ops
        else:
            merged_settings = self._settings
            other_custom_ops = {}

        if not merged_settings.operator_is_enabled(op_name):
            self._raise_type_error(op_name, other)

        self_custom_ops = self._settings.custom_ops
        merged_custom_ops = merged_settings.custom_ops

        if (
            op_name in self_custom_ops or op_name in other_custom_ops
        ) and op_name not in merged_custom_ops:
            self._raise_type_error(op_name, other, used_custom_op=True)

        other_func = other._func if isinstance(other, _OpFunc) else other

        if other_func is None:
            return merged_settings.get_unary_opfunc(
                op_name,
                self._func,
                *op_args,
                **op_kwargs,
            )

        return merged_settings.get_binary_opfunc(
            op_name,
            self._func,
            other_func,
            *op_args,
            **op_kwargs,
        )

    def _raise_type_error(
        self,
        op_name: str,
        other_func: _OpFunc | Func | None = None,
        used_custom_op: bool = False,
    ):
        self_name = self._func.__name__
        symbol = OPERATORS_DICT[op_name].symbol

        if used_custom_op:
            symbol = f"custom {symbol}"

        type_error_text = (
            f"unsupported operand type(s) for {symbol}: "
            f"{self.__class__.__name__!r} ({self_name!r})"
        )

        if other_func is not None:
            other_name = other_func.__name__
            cls_name = (
                other_func.__class__.__name__
                if isinstance(other_func, _OpFunc)
                else 'function'
            )
            type_error_text += f" and {cls_name!r} ({other_name!r})"

        raise TypeError(type_error_text)


class _OpFuncSettings(NamedTuple):
    include: tuple[str, ...] | None
    exclude: tuple[str, ...] | None
    custom_ops: dict[str, CustomOpBuilder]

    @classmethod
    def new(
        cls,
        include: str | Sequence[str] | None,
        exclude: str | Sequence[str] | None,
        custom_ops: Mapping[str, CustomOpBuilder] | None,
    ):
        include = _to_tuple(include)
        exclude = _to_tuple(exclude)

        _verify_mutually_exclusive(include, exclude)

        return cls(
            include=include,
            exclude=exclude,
            custom_ops=dict(custom_ops or {}),
        )

    def merge_settings(self, settings: _OpFuncSettings) -> _OpFuncSettings:
        include = _merge_tuples(self.include, settings.include, 'intersection')
        exclude = _merge_tuples(self.exclude, settings.exclude, 'union')
        custom_ops = _merge_dicts(self.custom_ops, settings.custom_ops)

        return _OpFuncSettings(
            include=include,
            exclude=exclude,
            custom_ops=custom_ops,
        )

    def operator_is_enabled(self, op_name: str) -> bool:
        if op_name in (self.exclude or ()):
            return False

        if self.include is None:
            return True

        return op_name in self.include

    def get_binary_opfunc(
        self,
        op_name: str,
        func1: Func,
        func2: Func,
        *op_args,
        **op_kwargs,
    ) -> _OpFunc[FuncParams, T]:
        if op_name in self.custom_ops:
            func_builder: BinaryOpBuilder = self.custom_ops[op_name]
            func = func_builder(func1, func2, *op_args, **op_kwargs)
        else:

            def binary_func(
                *args: FuncParams.args,
                **kwargs: FuncParams.kwargs,
            ) -> T:
                return OPERATORS_DICT[op_name].op(
                    func1(*args, **kwargs),
                    func2(*args, **kwargs),
                    *op_args,
                    **op_kwargs,
                )

            func = binary_func

        return _OpFunc[FuncParams, T](
            func,
            self,
        )

    def get_unary_opfunc(
        self,
        op_name: str,
        func: Func,
        *op_args: OpParams.args,
        **op_kwargs: OpParams.kwargs,
    ) -> _OpFunc[FuncParams, T]:
        if op_name in self.custom_ops.keys():
            func_builder: UnaryOpBuilder = self.custom_ops[op_name]
            func_ = func_builder(func, *op_args, **op_kwargs)
        else:

            def unary_func(
                *args: FuncParams.args,
                **kwargs: FuncParams.kwargs,
            ) -> T:
                return OPERATORS_DICT[op_name].op(
                    func(*args, **kwargs),
                    *op_args,
                    **op_kwargs,
                )

            func_ = unary_func

        return _OpFunc[FuncParams, T](
            func_,
            self,
        )


def _to_tuple(items: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if items is None:
        return None

    if isinstance(items, str):
        return (items,)

    return tuple(items)


def _merge_tuples(
    a: tuple[str, ...] | None,
    b: tuple[str, ...] | None,
    merge: Literal['union', 'intersection'],
) -> tuple[str, ...] | None:
    if a is None or b is None:
        return a if a is not None else b

    if merge == 'union':
        return tuple(set(a).union(b))

    if merge == 'intersection':
        return tuple(set(a).intersection(b))


def _merge_dicts(
    a: dict[str, CustomOpBuilder],
    b: dict[str, CustomOpBuilder],
) -> dict[str, CustomOpBuilder]:
    return {
        op_name: op_builder
        for op_name, op_builder in a.items()
        if b.get(op_name) == op_builder
    }


def _verify_mutually_exclusive(
    include: tuple[str, ...] | None,
    exclude: tuple[str, ...] | None,
):
    intersection = set(include or ()).intersection(exclude or ())

    if not intersection:
        return

    raise ValueError(
        f"cannot create opfunc with operators "
        f"{intersection!r} both included and excluded.",
    )

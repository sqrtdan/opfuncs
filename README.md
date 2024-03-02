# opfuncs
A package to enable the usage of operators on python functions.

```pycon
>>> from opfuncs import opfunc
>>> @opfunc
... def add_7(x):
...     return x + 7
...
>>> @opfunc
... def times_4(x):
...     return x * 4
...
>>> (add_7 + times_4)(5)
32
>>> (add_7 * times_4)(3)
120
>>> times_4[add_7](4)  # supports composition
44
```

---

# Installation
The opfuncs package can be installed from PyPI using pip:

```
pip install opfuncs
```

---

# Features
Can be used alongside normal functions occupying the right-side argument of the operator.
```pycon
>>> def line(x):
...     return 3 * x + 4
...
>>> (times_4 * line)(3)
156
```

---

You can restrict a function to only allow/disallow certain operators using the
`include` and `exclude` parameters:
```pycon
>>> @opfunc(include=('add', ))
>>> def mod_3(x):
...     return x % 3
```
Refer to the `SUPPORTED_OPERATORS` constant within the package for the list operator names supported by `include` and `exclude`.

---

You can customize an operator to use a function you define in place of the usual operator. Custom operators are supplied to the `opfunc` wrapper through the `custom_ops` parameter, which accepts a dictionary mapping operator names to operator function builders.
```pycon
>>> def custom_xor(
...     f: Callable[[float], float],
...     g: Callable[[float], float],
... ) -> Callable[[float], float]:
...     def xor_as_pow(x: float) -> float:
...         return f(x) ** g(x)
...
...     return xor_as_pow
...
>>> custom_times_4 = opfunc(times_4, custom_ops={"xor": custom_xor})
>>> custom_add_7 = opfunc(add_7, custom_ops={"xor": custom_xor})
>>> (custom_times_4 ^ custom_add_7)(-5)
400
```
Refer to the `SUPPORTED_OPERATORS` constant within the package for the list operator names supported by `custom_ops`.

---

# Supported Operators
The opfuncs package supports the following python operators and functions:
- `&`
- `|`
- `^`
- `+`
- `-`
- `*`
- `@`
- `/`
- `//`
- `%`
- `divmod()`
- `**` and `pow()`
- `<<`
- `>>`
- `-()`
- `+()`
- `abs()`
- `~`
- `round()`
- `math.trunc()`
- `math.floor()`
- `math.ceil()`

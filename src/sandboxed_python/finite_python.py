"""
Author: thautwarm <twshe@outlook.com>
License: BSD-3-Clause
Description:
    The module implements FPy (Finite Python), a lightweight, secure,
    and type-safe interpreter for a restricted subset of Python, designed
    specifically for improving LLM Tool Call capabilities in constrained
    environments. It provides a sandboxed execution environment that ensures
    safety while maintaining compatibility with Python's syntax and semantics.
    This module is ideal for scenarios where (strongly) controlled execution
    of Python code is required, such as in LLM tool calls, serverless functions,
    or plugin systems.
"""

from __future__ import annotations
from dataclasses import dataclass

import ast
import abc
import operator
import typing
import enum
import io
import traceback


# =============================================================================
# REGION: Core Types and Constants
# =============================================================================

type HashableFPyVal = (
    int
    | float
    | str
    | bool
    | None
    | tuple[HashableFPyVal, ...]
    | frozenset[HashableFPyVal]
    | slice
    | complex
    | bytes
)

type FPyVal = (
    int
    | float
    | str
    | bool
    | None
    | slice
    | bytes
    | complex
    | bytearray
    | list[FPyVal]
    | dict[HashableFPyVal, FPyVal]
    | set[HashableFPyVal]
    | tuple[FPyVal, ...]
)


def _assert_never(value: typing.Never) -> typing.Never:
    raise Exception(f"Unhandled value: {value}")


class Undef(enum.Enum):
    undef = enum.auto()


undef = Undef.undef


@dataclass(frozen=True)
class SourceLocation:
    """Enhanced source location with optional filename"""

    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    filename: str | None = None
    source: str | None = None

    def __str__(self) -> str:
        location = f"line {self.line}, column {self.column}"
        if self.filename:
            location = f"{self.filename}:{location}"
        return location

    def get_related_source(self, context_lines: int = 2) -> str:
        """
        Extract related source code with error indication.

        Example:
        --------------------------------------
                 1 def calculate(x, y):
            ---> 2     result = x + y
                                ^
                 3     if result > 10:
                 4         return result * 2
        --------------------------------------
        """
        if not self.source:
            return ""

        lines = self.source.splitlines()
        if not lines or self.line <= 0 or self.line > len(lines):
            return ""

        # We extract our own ASTs from Python `ast.AST`, which uses 1-based indexing.
        error_line_idx = self.line - 1

        # Determine the range of lines to show
        start_line = max(0, error_line_idx - context_lines)
        end_line = min(len(lines), error_line_idx + context_lines + 1)

        # Calculate the width needed for line numbers
        max_line_num = end_line
        line_num_width = len(str(max_line_num))

        result_lines: list[str] = []
        marker = "---> "
        ws_prefix = " " * len(marker)

        for i in range(start_line, end_line):
            line_num = i + 1
            line_content = lines[i]

            # Format line number with consistent width
            line_num_str = str(line_num).rjust(line_num_width)

            if i == error_line_idx:
                # Mark the error line with an arrow
                result_lines.append(f"{marker}{line_num_str} {line_content}")

                # Add error indicator line with ^^^
                if self.column > 0:
                    # Calculate the position for the error indicator
                    # Account for the line number prefix: "---> {line_num} "
                    prefix_len = (
                        len(marker) + line_num_width + 1
                    )  # "---> " + line_num + " "

                    # Calculate column position, handling tabs
                    actual_column = 0
                    for char in line_content[: self.column - 1]:
                        if char == "\t":
                            actual_column += 4  # Assume tab width of 4
                        else:
                            actual_column += 1

                    # Create the error indicator
                    spaces = " " * (prefix_len + actual_column)

                    # Determine the length of the error indicator
                    if self.end_column and self.end_line == self.line:
                        error_length = max(1, self.end_column - self.column)
                    else:
                        error_length = 1

                    caret_line = spaces + "^" * error_length
                    result_lines.append(caret_line)
            else:
                # Regular context line
                result_lines.append(f"{ws_prefix}{line_num_str} {line_content}")

        return "\n".join(result_lines)


# =============================================================================
# REGION: Error System (Type-Safe ADTs)
# =============================================================================
class ErrorSeverity(enum.Enum):
    """Error severity levels for better error handling"""

    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


@dataclass(frozen=True)
class FPyParseError:
    message: str
    location: SourceLocation
    suggestion: str | None = None


@dataclass(frozen=True)
class FPySimpleError:
    """
    Runtime error with a simple message and location.
    """

    message: str
    location: SourceLocation


@dataclass(frozen=True)
class FPyExecError:
    """
    Runtime error that comes from Python
    """

    error_type: str
    backtrace: str
    message: str
    location: SourceLocation


type FPyError = FPyParseError | FPySimpleError | FPyExecError


class _FPyExprInnerException(Exception):
    """
    Inner exception for expression evaluation errors.
    """

    def __init__(self, *, error_type: str, message: str, backtrace: str):
        self.error_type = error_type
        self.message = message
        self.backtrace = backtrace
        super().__init__("FPy expression evaluation error")


class FPyException(Exception):
    """Base exception with structured error information"""

    def __init__(self, error: FPyError):
        self.error = error
        super().__init__(FPyException.s_short_error_message(error))

    @classmethod
    def s_short_error_message(cls, error: FPyError) -> str:
        buf = io.StringIO()
        match error:
            case FPyParseError():
                buf.write(
                    f"<FPyParseError message={error.message!r}, location={error.location}>"
                )
            case FPySimpleError():
                buf.write(
                    f"<FPySimpleError message={error.message!r}, location={error.location}>"
                )
            case FPyExecError():
                buf.write(
                    f"<FPyExecError error_type={error.error_type!r}, message={error.message!r}, location={error.location}>"
                )
            case _unused:
                _assert_never(_unused)
        return buf.getvalue()

    def pretty_print(self) -> None:
        self.s_format(lambda x: print(x, end=""), self.error)

    def format(self, write: typing.Callable[[str], typing.Any]) -> None:
        self.s_format(write, self.error)

    @classmethod
    def s_format(cls, write: typing.Callable[[str], typing.Any], error: FPyError):
        """Formats the error into a human(llm)-readable string."""
        # buf = io.StringIO()
        match error:
            case FPyParseError():
                write(f"Parse error: {error.message} at {error.location}\n")
                write(error.location.get_related_source())
                write("\n")
                if error.suggestion:
                    write(f"Suggestion: {error.suggestion}")

            case FPySimpleError():
                write(f"Error: {error.message} at {error.location}\n")
                write(error.location.get_related_source())

            case FPyExecError():
                write(f"Exec error at {error.location}\n")
                write(error.location.get_related_source())
                write("\n")
                write("[Original Python Backtrace]\n")
                write(error.backtrace)

            case _unused:
                _assert_never(_unused)


# =============================================================================
# REGION: FPy AST Definition (Type-Safe ADTs)
# =============================================================================

type BoolOpType = typing.Literal["And", "Or"]
type BinOpType = typing.Literal[
    "Add",
    "Sub",
    "Mult",
    "Div",
    "FloorDiv",
    "Mod",
    "Pow",
    "LShift",
    "RShift",
    "BitOr",
    "BitXor",
    "BitAnd",
]
type UnaryOpType = typing.Literal["UAdd", "USub", "Not", "Invert"]
type CmpOpType = typing.Literal[
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn"
]
type Stmt = StmtExpr | StmtAssign | StmtIf | StmtDelete | StmtTry | StmtPass
type Expr = (
    ExprConst
    | ExprName
    | ExprAnd
    | ExprOr
    | ExprBin
    | ExprUnary
    | ExprCompare
    | ExprCall
    | ExprMethCall
    | ExprIndex
    | ExprList
    | ExprDict
    | ExprSet
    | ExprTuple
    | ExprSlice
)

type Lhs = LhsName | LhsIndex


# Expression nodes
@dataclass(frozen=True)
class ExprConst:
    value: FPyVal
    location: SourceLocation


@dataclass(frozen=True)
class ExprName:
    id: str
    location: SourceLocation


@dataclass(frozen=True)
class ExprAnd:
    left: Expr
    right: Expr
    location: SourceLocation


@dataclass(frozen=True)
class ExprOr:
    left: Expr
    right: Expr
    location: SourceLocation


@dataclass(frozen=True)
class ExprBin:
    left: Expr
    op: BinOpType
    right: Expr
    location: SourceLocation


@dataclass(frozen=True)
class ExprUnary:
    op: UnaryOpType
    operand: Expr
    location: SourceLocation


@dataclass(frozen=True)
class ExprCompare:
    left: Expr
    comparisons: list[tuple[CmpOpType, Expr]]
    location: SourceLocation


@dataclass(frozen=True)
class ExprCall:
    func: str  # Function name (identifier only)
    args: list[Expr]
    location: SourceLocation


@dataclass(frozen=True)
class ExprMethCall:
    subject: Expr
    method: str
    args: list[Expr]
    location: SourceLocation


@dataclass(frozen=True)
class ExprIndex:
    value: Expr
    index: Expr
    location: SourceLocation


@dataclass(frozen=True)
class ExprList:
    elements: list[Expr | StarredExpr]
    location: SourceLocation


@dataclass(frozen=True)
class ExprDict:
    keys: list[Expr | Undef]
    values: list[Expr]
    location: SourceLocation


@dataclass(frozen=True)
class StarredExpr:
    value: Expr
    location: SourceLocation


@dataclass(frozen=True)
class ExprSet:
    elements: list[Expr | StarredExpr]
    location: SourceLocation


@dataclass(frozen=True)
class ExprTuple:
    elements: list[Expr | StarredExpr]
    location: SourceLocation


@dataclass(frozen=True)
class ExprSlice:
    lower: Expr | None
    upper: Expr | None
    step: Expr | None
    location: SourceLocation


# Lhs nodes
@dataclass(frozen=True)
class LhsName:
    id: str
    location: SourceLocation


@dataclass(frozen=True)
class LhsIndex:
    subject: Expr
    index: Expr
    location: SourceLocation


# Statement nodes
@dataclass(frozen=True)
class StmtExpr:
    value: Expr
    location: SourceLocation


@dataclass(frozen=True)
class StmtPass:
    location: SourceLocation


@dataclass(frozen=True)
class StmtDelete:
    targets: list[Lhs]
    location: SourceLocation


@dataclass(frozen=True)
class StmtTry:
    body: list[Stmt]
    handler: TryHandler
    finally_body: list[Stmt]
    location: SourceLocation


@dataclass(frozen=True)
class TryHandler:
    name: str | None
    body: list[Stmt]
    location: SourceLocation


@dataclass(frozen=True)
class StmtAssign:
    targets: list[Lhs]  # identifier only
    value: Expr
    location: SourceLocation


@dataclass(frozen=True)
class StmtIf:
    test: Expr
    body: list[Stmt]
    orelse: list[Stmt]
    location: SourceLocation


# =============================================================================
# REGION: Interfaces
# =============================================================================


class PySandbox(abc.ABC):
    """Enhanced sandbox interface with security and monitoring"""

    @abc.abstractmethod
    def get_location(self) -> SourceLocation | None:
        """Get the current location of the sandbox"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_location(self, location: SourceLocation) -> None:
        """Set the current location of the sandbox"""
        raise NotImplementedError

    @abc.abstractmethod
    def func_call(
        self, func_name: str, args: list[FPyVal], location: SourceLocation
    ) -> FPyVal:
        """Execute a function call with location context"""
        raise NotImplementedError

    def bin_op(self, op: BinOpType, left: FPyVal, right: FPyVal) -> FPyVal:
        """Execute a binary operation with location context"""
        return _BIN_OP_HANDLERS[op](left, right)

    def cmp_op(self, op: CmpOpType, left: FPyVal, right: FPyVal) -> bool:
        """Execute a comparison operation with location context"""
        return _CMP_OP_HANDLERS[op](left, right)

    def unary_op(self, op: UnaryOpType, operand: FPyVal) -> FPyVal:
        """Execute a unary operation with location context"""
        return _UNARY_OP_HANDLERS[op](operand)

    @abc.abstractmethod
    def method_call(
        self,
        subject: FPyVal,
        method_name: str,
        args: list[FPyVal],
        location: SourceLocation,
    ) -> FPyVal:
        """Execute a method call with location context"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_var(self, name: str) -> FPyVal | Undef:
        """Get variable value with location context"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_var(self, name: str, value: FPyVal | Undef) -> None:
        """Set variable value with location context"""
        raise NotImplementedError

    @abc.abstractmethod
    def list_vars(self) -> typing.Iterable[str]:
        """List all variable names"""
        raise NotImplementedError

    @abc.abstractmethod
    def display(self, value: FPyVal) -> None:
        """Display value (for expression statements)"""
        raise NotImplementedError

    @abc.abstractmethod
    def exception_to_message(self, exception: Exception) -> str:
        raise NotImplementedError

    def format_backtrace(self, e) -> str:
        e.__traceback__ = None
        return traceback.format_exc()

    def on_error(self, error: FPyError) -> None:
        """Optional error handling hook for monitoring"""
        pass


# =============================================================================
# REGION: Parser (Enhanced with Better Error Handling)
# =============================================================================


def parse_fpy(source: str, filename: str | None = None) -> list[Stmt]:
    parser = _FPyParser(source, filename)
    try:
        result = parser._parse()
    except (UnicodeEncodeError, MemoryError, RecursionError):
        result = []
    return result


class _FPyParser:
    """Enhanced parser with comprehensive error handling"""

    def __init__(self, source: str, filename: str | None = None):
        self.filename = filename
        self.source = source

    def _parse(self) -> list[Stmt]:
        """Parse Python source into FPy statements"""
        try:
            tree = ast.parse(self.source, filename=self.filename or "<string>")
            return [self._parse_stmt(stmt) for stmt in tree.body]
        except SyntaxError as e:
            location = SourceLocation(
                line=e.lineno or 1,
                column=e.offset or 0,
                end_line=e.end_lineno,
                end_column=e.end_offset,
                filename=self.filename,
                source=self.source,
            )
            raise FPyException(
                FPyParseError(
                    message=f"Invalid Python syntax: {e.msg}",
                    location=location,
                    suggestion="Check for missing colons, brackets, or indentation",
                )
            )

    def _get_location(self, node: ast.AST) -> SourceLocation:
        """Extract enhanced location from AST node"""
        return SourceLocation(
            line=getattr(node, "lineno", 1),
            column=getattr(node, "col_offset", 0),
            end_line=getattr(node, "end_lineno", None),
            end_column=getattr(node, "end_col_offset", None),
            filename=self.filename,
            source=self.source,
        )

    def _parse_stmt(self, node: ast.stmt) -> Stmt:
        """Parse statement with comprehensive error handling"""
        location = self._get_location(node)

        match node:
            case ast.Expr(value=expr):
                return StmtExpr(value=self._parse_expr(expr), location=location)

            case ast.Assign(targets=targets, value=value):
                return StmtAssign(
                    targets=[self._parse_lhs(target) for target in targets],
                    value=self._parse_expr(value),
                    location=location,
                )

            case ast.Delete(targets=targets):
                return StmtDelete(
                    targets=[self._parse_lhs(target) for target in targets],
                    location=location,
                )

            case ast.If(test=test, body=body, orelse=orelse):
                return StmtIf(
                    test=self._parse_expr(test),
                    body=[self._parse_stmt(s) for s in body],
                    orelse=[self._parse_stmt(s) for s in orelse],
                    location=location,
                )

            case ast.Pass():
                # Skip pass statements
                return StmtPass(location=location)

            case ast.Try(body=body, handlers=handlers, finalbody=finalbody):
                if len(handlers) != 1:
                    raise FPyException(
                        FPyParseError(
                            message="Only one exception handler is supported",
                            location=location,
                        )
                    )
                handler = handlers[0]
                handler_location = self._get_location(handler)

                if handler.type is None:
                    return StmtTry(
                        body=[self._parse_stmt(s) for s in body],
                        handler=TryHandler(
                            name=handler.name,
                            body=[self._parse_stmt(s) for s in handler.body],
                            location=handler_location,
                        ),
                        finally_body=[self._parse_stmt(s) for s in finalbody],
                        location=location,
                    )
                # if 'type' is specified, it must be 'Exception'
                match handler.type:
                    case ast.Name(id="Exception"):
                        pass
                    case _:
                        exception_type = ast.unparse(handler.type)
                        raise FPyException(
                            FPyParseError(
                                message=f"Only 'Exception' is supported in FPy's exception handler, but got {exception_type}",
                                location=location,
                            )
                        )
                return StmtTry(
                    body=[self._parse_stmt(s) for s in body],
                    handler=TryHandler(
                        name=handler.name,
                        body=[self._parse_stmt(s) for s in handler.body],
                        location=handler_location,
                    ),
                    finally_body=[self._parse_stmt(s) for s in finalbody],
                    location=location,
                )
            case uncovered:
                raise FPyException(
                    FPyParseError(
                        message=f"Unsupported statement: {ast.unparse(uncovered)}",
                        location=location,
                        suggestion="FPy only supports assignments, if statements, and expressions",
                    )
                )

    def _parse_lhs(self, node: ast.expr) -> Lhs:
        """Parse left-hand side of assignment"""
        location = self._get_location(node)
        match node:
            case ast.Name(id=name):
                return LhsName(id=name, location=location)
            case ast.Subscript(value=value, slice=slice):
                return LhsIndex(
                    subject=self._parse_expr(value),
                    index=self._parse_expr(slice),
                    location=location,
                )
            case _:
                raise FPyException(
                    FPyParseError(
                        message="Invalid left-hand side of assignment",
                        location=location,
                    )
                )

    def _parse_expr(self, node: ast.expr) -> Expr:
        """Parse expression with comprehensive error handling"""
        location = self._get_location(node)

        match node:
            case ast.Constant(value=value):
                # CPython compiler guarantees that the value is a valid FPyVal
                # as they are all constants to compiler/marshal system.
                return ExprConst(value=value, location=location)  # type: ignore

            case ast.Name(id=name):
                return ExprName(id=name, location=location)

            case ast.BoolOp(op=op, values=values):
                if len(values) < 2:
                    raise FPyException(
                        FPyParseError(
                            message="Boolean operation requires at least 2 operands",
                            location=location,
                        )
                    )
                # Convert to right-associative binary operations
                result = self._parse_expr(values[-1])
                if isinstance(op, ast.And):
                    for value in reversed(values[:-1]):
                        result = ExprAnd(
                            left=self._parse_expr(value),
                            right=result,
                            location=location,
                        )
                else:
                    assert isinstance(op, ast.Or)
                    for value in values[:-1]:
                        result = ExprOr(
                            left=self._parse_expr(value),
                            right=result,
                            location=location,
                        )
                return result

            case ast.BinOp(left=left, op=op, right=right):
                return ExprBin(
                    left=self._parse_expr(left),
                    op=self._convert_binop(op),
                    right=self._parse_expr(right),
                    location=location,
                )

            case ast.UnaryOp(op=op, operand=operand):
                return ExprUnary(
                    op=self._convert_unaryop(op),
                    operand=self._parse_expr(operand),
                    location=location,
                )

            case ast.Compare(left=left, ops=ops, comparators=comparators):
                # Fixed implementation for chained comparisons
                if len(ops) != len(comparators):
                    raise FPyException(
                        FPyParseError(message="Malformed comparison", location=location)
                    )

                if not ops or not comparators:
                    raise FPyException(
                        FPyParseError(message="Malformed comparison", location=location)
                    )

                return ExprCompare(
                    left=self._parse_expr(left),
                    comparisons=[
                        (self._convert_cmpop(op), self._parse_expr(comparator))
                        for op, comparator in zip(ops, comparators)
                    ],
                    location=location,
                )

            case ast.Call(func=ast.Name(id=func_name), args=args, keywords=[]):
                return ExprCall(
                    func=func_name,
                    args=[self._parse_expr(arg) for arg in args],
                    location=location,
                )

            case ast.Call(
                func=ast.Attribute(value=subject, attr=method), args=args, keywords=[]
            ):
                return ExprMethCall(
                    subject=self._parse_expr(subject),
                    method=method,
                    args=[self._parse_expr(arg) for arg in args],
                    location=location,
                )

            case ast.Call():
                raise FPyException(
                    FPyParseError(
                        message="Complex function calls not supported",
                        location=location,
                        suggestion="Use simple function calls: func(args) or obj.method(args)",
                    )
                )

            case ast.Subscript(value=value, slice=slice):
                return ExprIndex(
                    value=self._parse_expr(value),
                    index=self._parse_expr(slice),
                    location=location,
                )

            case ast.List(elts=elements):
                list_elements: list[Expr | StarredExpr] = []
                for elt in elements:
                    if isinstance(elt, ast.Starred):
                        list_elements.append(
                            StarredExpr(
                                value=self._parse_expr(elt.value),
                                location=location,
                            )
                        )
                    else:
                        list_elements.append(self._parse_expr(elt))
                return ExprList(
                    elements=list_elements,
                    location=location,
                )

            case ast.Dict(keys=keys, values=values):
                if len(keys) != len(values):
                    raise FPyException(
                        FPyParseError(
                            message="Dictionary keys and values mismatch",
                            location=location,
                        )
                    )

                dict_keys: list[Expr | Undef] = []
                for key in keys:
                    if key is None:
                        dict_keys.append(Undef.undef)
                    else:
                        dict_keys.append(self._parse_expr(key))

                return ExprDict(
                    keys=dict_keys,
                    values=[self._parse_expr(value) for value in values],
                    location=location,
                )

            case ast.Set(elts=elements):
                set_elements: list[Expr | StarredExpr] = []
                for elt in elements:
                    if isinstance(elt, ast.Starred):
                        set_elements.append(
                            StarredExpr(
                                value=self._parse_expr(elt.value),
                                location=location,
                            )
                        )
                    else:
                        set_elements.append(self._parse_expr(elt))
                return ExprSet(
                    elements=set_elements,
                    location=location,
                )
            case ast.Tuple(elts=elements):
                tuple_elements: list[Expr | StarredExpr] = []
                for elt in elements:
                    if isinstance(elt, ast.Starred):
                        tuple_elements.append(
                            StarredExpr(
                                value=self._parse_expr(elt.value),
                                location=location,
                            )
                        )
                    else:
                        tuple_elements.append(self._parse_expr(elt))
                return ExprTuple(
                    elements=tuple_elements,
                    location=location,
                )

            case ast.Slice(lower=lower, upper=upper, step=step):
                if lower is not None:
                    lower = self._parse_expr(lower)
                if upper is not None:
                    upper = self._parse_expr(upper)
                if step is not None:
                    step = self._parse_expr(step)
                return ExprSlice(
                    lower=lower,
                    upper=upper,
                    step=step,
                    location=location,
                )
            case _:
                raise FPyException(
                    FPyParseError(
                        message=f"Unsupported expression: {type(node).__name__}",
                        location=location,
                        suggestion="Check FPy documentation for supported expressions",
                    )
                )

    def _convert_boolop(self, op: ast.boolop) -> BoolOpType:
        match op:
            case ast.And():
                return "And"
            case ast.Or():
                return "Or"
            case _:
                raise ValueError(f"Unknown boolean operator: {type(op)}")

    def _convert_binop(self, op: ast.operator) -> BinOpType:
        match op:
            case ast.Add():
                return "Add"
            case ast.Sub():
                return "Sub"
            case ast.Mult():
                return "Mult"
            case ast.Div():
                return "Div"
            case ast.FloorDiv():
                return "FloorDiv"
            case ast.Mod():
                return "Mod"
            case ast.Pow():
                return "Pow"
            case ast.LShift():
                return "LShift"
            case ast.RShift():
                return "RShift"
            case ast.BitOr():
                return "BitOr"
            case ast.BitXor():
                return "BitXor"
            case ast.BitAnd():
                return "BitAnd"
            case _:
                raise ValueError(f"Unknown binary operator: {type(op).__name__}")

    def _convert_unaryop(self, op: ast.unaryop) -> UnaryOpType:
        match op:
            case ast.UAdd():
                return "UAdd"
            case ast.USub():
                return "USub"
            case ast.Not():
                return "Not"
            case ast.Invert():
                return "Invert"
            case _:
                raise ValueError(f"Unknown unary operator: {type(op)}")

    def _convert_cmpop(self, op: ast.cmpop) -> CmpOpType:
        match op:
            case ast.Eq():
                return "Eq"
            case ast.NotEq():
                return "NotEq"
            case ast.Lt():
                return "Lt"
            case ast.LtE():
                return "LtE"
            case ast.Gt():
                return "Gt"
            case ast.GtE():
                return "GtE"
            case ast.Is():
                return "Is"
            case ast.IsNot():
                return "IsNot"
            case ast.In():
                return "In"
            case ast.NotIn():
                return "NotIn"
            case _:
                raise ValueError(f"Unknown comparison operator: {type(op)}")


# =============================================================================
# REGION: Interpreter (Enhanced with Security and Performance)
# =============================================================================


def _op_in(a, b):
    return a in b


def _op_not_in(a, b):
    return a not in b


_UNARY_OP_HANDLERS: dict[UnaryOpType, typing.Callable[[typing.Any], typing.Any]] = {
    "Not": operator.not_,
    "Invert": operator.invert,
    "UAdd": operator.pos,
    "USub": operator.neg,
}

_CMP_OP_HANDLERS: dict[CmpOpType, typing.Callable[[typing.Any, typing.Any], bool]] = {
    "Eq": operator.eq,
    "NotEq": operator.ne,
    "Lt": operator.lt,
    "LtE": operator.le,
    "Gt": operator.gt,
    "GtE": operator.ge,
    "Is": operator.is_,
    "IsNot": operator.is_not,
    "In": _op_in,
    "NotIn": _op_not_in,
}

_BIN_OP_HANDLERS: dict[
    BinOpType, typing.Callable[[typing.Any, typing.Any], typing.Any]
] = {
    "Add": operator.add,
    "Sub": operator.sub,
    "Mult": operator.mul,
    "Div": operator.truediv,
    "FloorDiv": operator.floordiv,
    "Mod": operator.mod,
    "Pow": operator.pow,
    "LShift": operator.lshift,
    "RShift": operator.rshift,
    "BitOr": operator.or_,
    "BitXor": operator.xor,
    "BitAnd": operator.and_,
}


def exec_stmts(sandbox: PySandbox, statements: list[Stmt]) -> None:
    """Execute FPy statements with comprehensive error handling"""
    for stmt in statements:
        try:
            _execute_stmt(sandbox, stmt)
        except FPyException:
            raise
        except Exception as e:
            location = stmt.location
            error = FPyExecError(
                error_type=type(e).__name__,
                message=sandbox.exception_to_message(e),
                backtrace=sandbox.format_backtrace(e),
                location=location,
            )
            sandbox.on_error(error)
            raise FPyException(error)


def _execute_stmt(sandbox: PySandbox, stmt: Stmt) -> None:
    """Execute a single statement with type-safe pattern matching"""
    sandbox.set_location(stmt.location)
    try:
        match stmt:
            case StmtExpr(value=expr):
                result = _evaluate_expr(sandbox, expr)
                sandbox.display(result)

            case StmtAssign(targets=targets, value=expr):
                value = _evaluate_expr(sandbox, expr)
                for target in targets:
                    match target:
                        case LhsName(id=name):
                            sandbox.set_var(name, value)
                        case LhsIndex(subject=subject, index=index):
                            lhs_subject = _evaluate_expr(sandbox, subject)
                            lhs_index = _evaluate_expr(sandbox, index)
                            lhs_subject[lhs_index] = value  # type: ignore

            case StmtIf(test=test, body=body, orelse=orelse):
                test_result = _evaluate_expr(sandbox, test)
                if test_result:
                    for body_stmt in body:
                        _execute_stmt(sandbox, body_stmt)
                else:
                    for else_stmt in orelse:
                        _execute_stmt(sandbox, else_stmt)
            case StmtDelete(targets=targets):
                for target in targets:
                    match target:
                        case LhsName(id=name):
                            sandbox.set_var(name, undef)
                        case LhsIndex(subject=subject, index=index):
                            lhs_subject = _evaluate_expr(sandbox, subject)
                            lhs_index = _evaluate_expr(sandbox, index)
                            del lhs_subject[lhs_index]  # type: ignore
                        case _unused:
                            _assert_never(_unused)
            case StmtPass():
                pass

            case StmtTry(body=body, handler=handler, finally_body=finally_body):
                _execute_try(sandbox, body, handler, finally_body)

            case uncovered:
                _assert_never(uncovered)
    except FPyException:
        raise
    except Exception as e:
        location = stmt.location
        raise FPyException(
            FPyExecError(
                error_type=type(e).__name__,
                message=sandbox.exception_to_message(e),
                backtrace=sandbox.format_backtrace(e),
                location=location,
            )
        )


def _execute_try(
    sandbox: PySandbox, body: list[Stmt], handler: TryHandler, finally_body: list[Stmt]
) -> None:
    try:
        for stmt in body:
            _execute_stmt(sandbox, stmt)
    except FPyException as e:
        inner_error = e.error
        if handler.name is None:
            for stmt in handler.body:
                _execute_stmt(sandbox, stmt)
        else:
            dict_error: dict[HashableFPyVal, FPyVal]
            match inner_error:
                case FPyParseError():
                    raise
                case FPySimpleError():
                    dict_error = {
                        "type": "SimpleError",
                        "message": inner_error.message,
                        "py_stacktrace": "",
                    }
                case FPyExecError():
                    dict_error = {
                        "type": inner_error.error_type,
                        "message": inner_error.message,
                        "py_stacktrace": inner_error.backtrace,
                    }
                case _:
                    _assert_never(inner_error)

            sandbox.set_var(handler.name, dict_error)
            for stmt in handler.body:
                _execute_stmt(sandbox, stmt)

    finally:
        for stmt in finally_body:
            _execute_stmt(sandbox, stmt)


def _evaluate_expr(sandbox: PySandbox, expr: Expr) -> FPyVal:
    """Evaluate expression with comprehensive error handling"""
    try:
        sandbox.set_location(expr.location)
        match expr:
            case ExprConst(value=value):
                return value

            case ExprName(id=name, location=location):
                result = sandbox.get_var(name)
                if isinstance(result, Undef):
                    raise FPyException(
                        FPySimpleError(
                            message=f"Name '{name}' is not defined",
                            location=location,
                        )
                    )
                return result

            case ExprAnd(left=left, right=right):
                left_val = _evaluate_expr(sandbox, left)
                if not left_val:
                    return left_val
                return _evaluate_expr(sandbox, right)

            case ExprOr(left=left, right=right):
                left_val = _evaluate_expr(sandbox, left)
                if left_val:
                    return left_val
                return _evaluate_expr(sandbox, right)

            case ExprBin(left=left, op=op, right=right):
                return _evaluate_binop(sandbox, left, op, right)

            case ExprUnary(op=op, operand=operand):
                return _evaluate_unaryop(sandbox, op, operand)

            case ExprCompare(left=left, comparisons=comparisons):
                return _eval_compare(sandbox, left, comparisons)

            case ExprCall(func=func, args=args, location=location):
                arg_values = [_evaluate_expr(sandbox, arg) for arg in args]
                return sandbox.func_call(func, arg_values, location)

            case ExprMethCall(
                subject=subject, method=method, args=args, location=location
            ):
                subject_value = _evaluate_expr(sandbox, subject)
                arg_values = [_evaluate_expr(sandbox, arg) for arg in args]
                return sandbox.method_call(subject_value, method, arg_values, location)

            case ExprIndex(value=value, index=index):
                return _evaluate_subscript(sandbox, value, index)

            case ExprList(elements=elements):
                return _eval_list(sandbox, elements)

            case ExprDict(keys=keys, values=values):
                return _eval_dict(sandbox, keys, values)

            case ExprSet(elements=elements):
                return _eval_set(sandbox, elements)

            case ExprSlice(lower=lower, upper=upper, step=step):
                return _eval_slice(sandbox, lower, upper, step)

            case ExprTuple(elements=elements):
                return _eval_tuple(sandbox, elements)

            case _unused:
                _assert_never(_unused)

    except FPyException:
        raise

    # We use 2-stage exception handling to avoid
    # the need to pass the location to every function.
    except _FPyExprInnerException as e:
        e.__traceback__ = None
        raise FPyException(
            FPyExecError(
                error_type=e.error_type,
                message=e.message,
                backtrace=e.backtrace,
                location=expr.location,
            )
        )

    except Exception as e:
        location = expr.location
        raise FPyException(
            FPyExecError(
                error_type=type(e).__name__,
                message=sandbox.exception_to_message(e),
                backtrace=sandbox.format_backtrace(e),
                location=location,
            )
        )


def _eval_list(sandbox: PySandbox, elements: list[Expr | StarredExpr]) -> list[FPyVal]:
    list_result: list[FPyVal] = []
    for element in elements:
        if isinstance(element, StarredExpr):
            list_result.extend(_evaluate_expr(sandbox, element.value))  # type: ignore
        else:
            list_result.append(_evaluate_expr(sandbox, element))
    return list_result


def _eval_dict(
    sandbox: PySandbox, keys: list[Expr | Undef], values: list[Expr]
) -> dict[HashableFPyVal, FPyVal]:
    dict_result: dict[HashableFPyVal, FPyVal] = {}
    for key_expr, val_expr in zip(keys, values):
        if isinstance(key_expr, Undef):
            vals = _evaluate_expr(sandbox, val_expr)  # type: ignore
            dict_result.update(vals)  # type: ignore
            continue
        val = _evaluate_expr(sandbox, val_expr)
        key = _evaluate_expr(sandbox, key_expr)
        dict_result[key] = val  # type: ignore
    return dict_result


def _eval_set(
    sandbox: PySandbox, elements: list[Expr | StarredExpr]
) -> set[HashableFPyVal]:
    set_result: set[HashableFPyVal] = set()
    elt_val: typing.Any  # erase the value as .add method will check the hashable type
    for element in elements:
        if isinstance(element, StarredExpr):
            elt_val = _evaluate_expr(sandbox, element.value)
            set_result.update(elt_val)
        else:
            elt_val = _evaluate_expr(sandbox, element)
            set_result.add(elt_val)
    return set_result


def _eval_slice(
    sandbox: PySandbox, lower: Expr | None, upper: Expr | None, step: Expr | None
) -> FPyVal:
    if lower is not None:
        lower = _evaluate_expr(sandbox, lower)  # type: ignore
    if upper is not None:
        upper = _evaluate_expr(sandbox, upper)  # type: ignore
    if step is not None:
        step = _evaluate_expr(sandbox, step)  # type: ignore

    return slice(lower, upper, step)


def _eval_tuple(
    sandbox: PySandbox, elements: list[Expr | StarredExpr]
) -> tuple[FPyVal, ...]:
    tuple_result: list[FPyVal] = []
    for element in elements:
        if isinstance(element, StarredExpr):
            tuple_result.extend(_evaluate_expr(sandbox, element.value))  # type: ignore
        else:
            tuple_result.append(_evaluate_expr(sandbox, element))
    return tuple(tuple_result)


def _eval_compare(
    sandbox: PySandbox,
    left: Expr,
    comparisons: list[tuple[CmpOpType, Expr]],
) -> bool:
    left_val = _evaluate_expr(sandbox, left)
    for op, right in comparisons:
        right_val = _evaluate_expr(sandbox, right)
        if not sandbox.cmp_op(op, left_val, right_val):
            return False
        left_val = right_val
    return True


def _evaluate_binop(
    sandbox: PySandbox, left: Expr, op: BinOpType, right: Expr
) -> FPyVal:
    """Evaluate binary operation with error handling"""
    left_val = _evaluate_expr(sandbox, left)
    right_val = _evaluate_expr(sandbox, right)

    try:
        return sandbox.bin_op(op, left_val, right_val)
    except Exception as e:
        e.__traceback__ = None
        raise _FPyExprInnerException(
            error_type=type(e).__name__,
            message=sandbox.exception_to_message(e),
            backtrace=sandbox.format_backtrace(e),
        )


def _evaluate_unaryop(sandbox: PySandbox, op: UnaryOpType, operand: Expr) -> FPyVal:
    """Evaluate unary operation with error handling"""
    operand_val = _evaluate_expr(sandbox, operand)

    try:
        if op == "Not":
            return not operand_val
        else:
            return sandbox.unary_op(op, operand_val)
    except Exception as e:
        raise _FPyExprInnerException(
            error_type=type(e).__name__,
            message=sandbox.exception_to_message(e),
            backtrace=sandbox.format_backtrace(e),
        )


def _evaluate_subscript(sandbox: PySandbox, value: Expr, index: Expr) -> FPyVal:
    """Evaluate subscript operation with error handling"""
    value_val = _evaluate_expr(sandbox, value)
    index_val = _evaluate_expr(sandbox, index)

    try:
        return value_val[index_val]  # type: ignore
    except Exception as e:
        raise _FPyExprInnerException(
            error_type=type(e).__name__,
            message=sandbox.exception_to_message(e),
            backtrace=sandbox.format_backtrace(e),
        )


# =============================================================================
# REGION: Secure Sandbox Implementation
# =============================================================================


class DefaultSecureSandbox(PySandbox):
    """Production-ready secure sandbox with comprehensive protections"""

    def __init__(
        self,
        allowed_builtins: dict[str, typing.Callable] | None = None,
        max_recursion_depth: int = 100,
        enable_monitoring: bool = True,
    ):
        self.variables: dict[str, FPyVal] = {}
        self.call_stack: list[str] = []
        self.max_recursion_depth = max_recursion_depth
        self.enable_monitoring = enable_monitoring
        self.location: SourceLocation | None = None

        # Default safe builtins
        self.allowed_builtins = allowed_builtins or {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "reversed": lambda x: list(reversed(x)),
            "print": self._safe_print,
        }

        self.allowed_methods = {
            str: {
                "upper": str.upper,
                "lower": str.lower,
                "strip": str.strip,
                "split": str.split,
                "join": str.join,
                "replace": str.replace,
                "startswith": str.startswith,
                "endswith": str.endswith,
                "find": str.find,
            },
            list: {
                "append": list.append,
                "extend": list.extend,
                "insert": list.insert,
                "remove": list.remove,
                "pop": list.pop,
                "index": list.index,
                "count": list.count,
                "sort": list.sort,
                "reverse": list.reverse,
                "copy": list.copy,
            },
            dict: {
                "get": dict.get,
                "keys": lambda self: list(self.keys()),
                "values": lambda self: list(self.values()),
                "items": lambda self: list(self.items()),
                "copy": dict.copy,
                "update": dict.update,
                "pop": dict.pop,
                "clear": dict.clear,
            },
            set: {
                "add": set.add,
                "remove": set.remove,
                "pop": set.pop,
                "clear": set.clear,
                "copy": set.copy,
                "union": set.union,
                "intersection": set.intersection,
                "difference": set.difference,
                "symmetric_difference": set.symmetric_difference,
                "issubset": set.issubset,
                "issuperset": set.issuperset,
                "isdisjoint": set.isdisjoint,
            },
        }

    def get_location(self) -> SourceLocation | None:
        """Get the current location of the sandbox"""
        return self.location

    def set_location(self, location: SourceLocation) -> None:
        """Set the current location of the sandbox"""
        self.location = location

    def _safe_print(self, *args: FPyVal) -> None:
        """Safe print implementation"""
        print("[FPy Output]", *args)
        return None

    def _check_recursion_depth(self, location: SourceLocation) -> None:
        """Check for excessive recursion"""
        if len(self.call_stack) > self.max_recursion_depth:
            raise FPyException(
                FPySimpleError(
                    message=f"Maximum recursion depth exceeded ({self.max_recursion_depth})",
                    location=location,
                )
            )

    def func_call(
        self, func_name: str, args: list[FPyVal], location: SourceLocation
    ) -> typing.Any:
        """Secure function call with monitoring"""
        self._check_recursion_depth(location)

        if func_name not in self.allowed_builtins:
            raise FPyException(
                FPySimpleError(
                    message=f"Function '{func_name}' is not allowed or defined",
                    location=location,
                )
            )

        try:
            self.call_stack.append(f"func:{func_name}")
            if self.enable_monitoring:
                print(f"[MONITOR] Calling function: {func_name} with {len(args)} args")

            result = self.allowed_builtins[func_name](*args)
            return result

        except FPyException:
            raise
        except Exception as e:
            raise FPyException(
                FPyExecError(
                    error_type=type(e).__name__,
                    message=self.exception_to_message(e),
                    backtrace=self.format_backtrace(e),
                    location=location,
                )
            )
        finally:
            if self.call_stack:
                self.call_stack.pop()

    def method_call(
        self,
        subject: FPyVal,
        method_name: str,
        args: list[FPyVal],
        location: SourceLocation,
    ) -> typing.Any:
        """Secure method call with comprehensive safety checks"""
        self._check_recursion_depth(location)

        subject_type = type(subject)
        if (vtable := self.allowed_methods.get(subject_type)) and (
            meth := vtable.get(method_name)
        ):
            pass
        else:
            raise FPyException(
                FPySimpleError(
                    message=f"Method '{method_name}' not allowed on {subject_type.__name__}",
                    location=location,
                )
            )

        try:
            self.call_stack.append(f"method:{subject_type.__name__}.{method_name}")
            if self.enable_monitoring:
                print(
                    f"[MONITOR] Calling method: {subject_type.__name__}.{method_name} with {len(args)} args"
                )
            if isinstance(meth, staticmethod):
                result = meth.__func__(*args)
            elif isinstance(meth, classmethod):
                result = meth.__func__(subject_type, *args)  # type: ignore
            else:
                result = meth(subject, *args)

            return result

        except FPyException:
            raise
        except Exception as e:
            raise FPyException(
                FPyExecError(
                    error_type=type(e).__name__,
                    message=self.exception_to_message(e),
                    backtrace=self.format_backtrace(e),
                    location=location,
                )
            )
        finally:
            if self.call_stack:
                self.call_stack.pop()

    def get_var(self, name: str) -> FPyVal | Undef:
        """Get variable with monitoring"""
        if self.enable_monitoring and name not in self.variables:
            print(f"[MONITOR] Accessing undefined variable: {name}")

        return self.variables.get(name, undef)

    def list_vars(self) -> typing.Iterable[str]:
        """List all variable names"""
        return self.variables.keys()

    def set_var(self, name: str, value: FPyVal | Undef) -> None:
        if self.enable_monitoring:
            print(f"[MONITOR] Setting variable: {name} = {type(value).__name__}")

        if value is undef:
            self.variables.pop(name, None)
            return

        self.variables[name] = value  # type: ignore

    def display(self, value: FPyVal) -> None:
        """Display value with safe formatting"""
        if self.enable_monitoring:
            print(f"[MONITOR] Displaying: {type(value).__name__}")

        if value is None:
            return

        try:
            print(f"[FPy Result] {value!r}")
        except Exception:
            print(f"[FPy Result] <unprintable {type(value).__name__}>")

    def on_error(self, error: FPyError) -> None:
        """Error monitoring hook"""
        if self.enable_monitoring:
            print(f"[MONITOR] Error occurred: {type(error).__name__} - {error}")

    def exception_to_message(self, exception: Exception) -> str:
        return str(exception)

    def add_function(self, name: str, func: typing.Callable) -> None:
        """Add a function to the sandbox"""
        self.allowed_builtins[name] = func

    def add_method(
        self, type_name: str, method_name: str, method: typing.Callable
    ) -> None:
        """Add a method to the sandbox"""
        self.allowed_methods[type_name][method_name] = method


# =============================================================================
# REGION: High-Level API and Utilities
# =============================================================================


def execute_fpy(
    source: str,
    sandbox: PySandbox | None = None,
    *,
    ast: list[Stmt] | None = None,
    filename: str | None = None,
    enable_monitoring: bool = False,
) -> None:
    """High-level function to parse and execute FPy code"""
    if sandbox is None:
        sandbox = DefaultSecureSandbox(enable_monitoring=enable_monitoring)

    if ast is None:
        statements = parse_fpy(source, filename=filename)
    else:
        statements = ast
    exec_stmts(sandbox, statements)


# Export main components for library usage
__all__ = [
    # Core types
    "FPyVal",
    "HashableFPyVal",
    "Undef",
    "undef",
    "SourceLocation",
    "PySandbox",
    "DefaultSecureSandbox",
    "parse_fpy",
    # Error types
    "FPyError",
    "FPyException",
    "FPyParseError",
    "FPySimpleError",
    "FPyExecError",
    # High-level API
    "execute_fpy",
]

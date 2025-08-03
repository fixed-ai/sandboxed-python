from __future__ import annotations
import unittest
from unittest.mock import patch
import io
from sandboxed_python import (
    parse_fpy,
    execute_fpy,
    SourceLocation,
    DefaultSecureSandbox,
    FPyException,
    FPyParseError,
    FPySimpleError,
    FPyExecError,
)
from sandboxed_python.finite_python import (
    ExprConst,
    ExprName,
    ExprBin,
    ExprList,
    ExprDict,
    ExprSet,
    ExprTuple,
    ExprSlice,
    ExprIndex,
    StarredExpr,
    StmtExpr,
    StmtAssign,
    StmtIf,
    StmtTry,
    StmtDelete,
    StmtPass,
    Undef,
    undef,
    LhsName,
    exec_stmts,
    _evaluate_expr,
)

# 假设您的代码文件名为 fpy.py（或直接导入）
# from fpy import *  # 替换为实际导入
# 为简洁，我在这里模拟导入；实际中替换为真实导入
# 注意：测试假设基于我的修复建议已应用（e.g., And 逻辑、星号展开、变量遮蔽）

# ... (在这里粘贴您的整个代码，或使用 import)

# 如果是独立文件，确保导入所有必要组件
# 为演示，我假设所有类/函数已定义。


class TestParser(unittest.TestCase):
    """测试解析器 (_FPyParser 和 parse_fpy)"""

    def test_parse_simple_expr(self):
        source = "1 + 2"
        stmts = parse_fpy(source)
        self.assertIsInstance(stmts[0], StmtExpr)
        self.assertIsInstance(stmts[0].value, ExprBin)
        self.assertEqual(stmts[0].value.op, "Add")

    def test_parse_assign(self):
        source = "x = 1"
        stmts = parse_fpy(source)
        self.assertIsInstance(stmts[0], StmtAssign)
        self.assertIsInstance(stmts[0].targets[0], LhsName)
        self.assertEqual(stmts[0].targets[0].id, "x")
        self.assertIsInstance(stmts[0].value, ExprConst)
        self.assertEqual(stmts[0].value.value, 1)

    def test_parse_if(self):
        source = """
if True:
    x = 1
else:
    x = 2
"""
        stmts = parse_fpy(source)
        self.assertIsInstance(stmts[0], StmtIf)
        self.assertTrue(len(stmts[0].body) == 1)
        self.assertTrue(len(stmts[0].orelse) == 1)

    def test_parse_try(self):
        source = """
try:
    x = 1 / 0
except Exception as e:
    pass
finally:
    print('done')
"""
        stmts = parse_fpy(source)
        self.assertIsInstance(stmts[0], StmtTry)
        self.assertEqual(stmts[0].handler.name, "e")
        self.assertTrue(len(stmts[0].finally_body) > 0)

    def test_parse_delete(self):
        source = "del x"
        stmts = parse_fpy(source)
        self.assertIsInstance(stmts[0], StmtDelete)
        self.assertEqual(stmts[0].targets[0].id, "x")

    def test_parse_pass(self):
        source = "pass"
        stmts = parse_fpy(source)
        self.assertIsInstance(stmts[0], StmtPass)

    def test_parse_list_with_starred(self):
        source = "[1, * [2,3], 4]"
        stmts = parse_fpy(source)
        expr = stmts[0].value  # type: ignore
        self.assertIsInstance(expr, ExprList)
        self.assertIsInstance(expr.elements[1], StarredExpr)

    def test_parse_dict_with_undef_key(self):
        source = "{** {'a':1}, 'b':2}"
        stmts = parse_fpy(source)
        expr = stmts[0].value  # type: ignore
        self.assertIsInstance(expr, ExprDict)
        self.assertIsInstance(expr.keys[0], Undef)

    def test_parse_set_with_starred(self):
        source = "{1, * [2,3]}"
        stmts = parse_fpy(source)
        expr = stmts[0].value  # type: ignore
        self.assertIsInstance(expr, ExprSet)
        self.assertIsInstance(expr.elements[1], StarredExpr)

    def test_parse_tuple_with_starred(self):
        source = "(1, * (2,3))"
        stmts = parse_fpy(source)
        expr = stmts[0].value  # type: ignore
        self.assertIsInstance(expr, ExprTuple)
        self.assertIsInstance(expr.elements[1], StarredExpr)

    def test_parse_slice(self):
        source = "lst[1:3:2]"
        stmts = parse_fpy(source)
        expr = stmts[0].value  # type: ignore
        self.assertIsInstance(expr, ExprIndex)
        self.assertIsInstance(expr.index, ExprSlice)

        source = "lst[:]"
        stmts = parse_fpy(source)
        expr = stmts[0].value  # type: ignore
        self.assertIsInstance(expr, ExprIndex)
        self.assertIsInstance(expr.index, ExprSlice)
        self.assertIsNone(expr.index.lower)
        self.assertIsNone(expr.index.upper)
        self.assertIsNone(expr.index.step)

    def test_parse_error_unsupported_stmt(self):
        source = "def f(): pass"
        with self.assertRaises(FPyException) as cm:
            parse_fpy(source)
        err = cm.exception.error
        self.assertIsInstance(err, FPyParseError)
        self.assertIn("Unsupported statement", err.message)

    def test_parse_error_syntax(self):
        source = "x = 1 + "
        with self.assertRaises(FPyException) as cm:
            parse_fpy(source)
        err = cm.exception.error
        self.assertIsInstance(err, FPyParseError)
        self.assertIn("Invalid Python syntax", err.message)

    def test_parse_error_multiple_handlers(self):
        source = """
try:
    pass
except TypeError:
    pass
except ValueError:
    pass
"""
        with self.assertRaises(FPyException) as cm:
            parse_fpy(source)
        err = cm.exception.error
        self.assertIn("Only one exception handler", err.message)

    def test_parse_error_invalid_lhs(self):
        source = "(x, y) = 1"
        with self.assertRaises(FPyException) as cm:
            parse_fpy(source)
        err = cm.exception.error
        self.assertIn("Invalid left-hand side", err.message)

    def test_parse_error_complex_call(self):
        source = "func(a=b)"
        with self.assertRaises(FPyException) as cm:
            parse_fpy(source)
        err = cm.exception.error
        self.assertIn("Complex function calls not supported", err.message)


class TestInterpreter(unittest.TestCase):
    """测试解释器 (exec_stmts, _execute_stmt, _evaluate_expr)"""

    def setUp(self):
        self.sandbox = DefaultSecureSandbox(enable_monitoring=False)
        self.sandbox.add_function("identity", lambda x: x)  # 测试用函数

    def test_execute_assign_and_expr(self):
        source = """
x = 1 + 2
x
"""
        stmts = parse_fpy(source)
        exec_stmts(self.sandbox, stmts)
        self.assertEqual(self.sandbox.get_var("x"), 3)

    def test_execute_if_true(self):
        source = """
if True:
    x = 1
else:
    x = 2
"""
        stmts = parse_fpy(source)
        exec_stmts(self.sandbox, stmts)
        self.assertEqual(self.sandbox.get_var("x"), 1)

    def test_execute_if_false(self):
        source = """
if False:
    x = 1
else:
    x = 2
"""
        stmts = parse_fpy(source)
        exec_stmts(self.sandbox, stmts)
        self.assertEqual(self.sandbox.get_var("x"), 2)

    def test_execute_try_no_error(self):
        source = """
try:
    x = 1
except Exception as e:
    x = 2
finally:
    y = 3
"""
        stmts = parse_fpy(source)
        exec_stmts(self.sandbox, stmts)
        self.assertEqual(self.sandbox.get_var("x"), 1)
        self.assertEqual(self.sandbox.get_var("y"), 3)

    def test_execute_try_with_error(self):
        source = """
try:
    1 / 0
except Exception as e:
    error_type = e['type']
finally:
    y = 3
"""
        stmts = parse_fpy(source)
        exec_stmts(self.sandbox, stmts)
        self.assertEqual(self.sandbox.get_var("error_type"), "ZeroDivisionError")
        self.assertEqual(self.sandbox.get_var("y"), 3)

    def test_execute_delete_var(self):
        self.sandbox.set_var("x", 1)
        source = "del x"
        stmts = parse_fpy(source)
        exec_stmts(self.sandbox, stmts)
        self.assertIsInstance(self.sandbox.get_var("x"), Undef)

    def test_execute_delete_index(self):
        self.sandbox.set_var("lst", [1, 2, 3])
        source = "del lst[1]"
        stmts = parse_fpy(source)
        exec_stmts(self.sandbox, stmts)
        self.assertEqual(self.sandbox.get_var("lst"), [1, 3])

    def test_execute_pass(self):
        source = "pass"
        stmts = parse_fpy(source)
        exec_stmts(self.sandbox, stmts)  # 无异常即可

    def test_evaluate_const(self):
        expr = ExprConst(42, SourceLocation(1, 1))
        self.assertEqual(_evaluate_expr(self.sandbox, expr), 42)

    def test_evaluate_name_defined(self):
        self.sandbox.set_var("x", 42)
        expr = ExprName("x", SourceLocation(1, 1))
        self.assertEqual(_evaluate_expr(self.sandbox, expr), 42)

    def test_evaluate_name_undefined(self):
        expr = ExprName("undefined", SourceLocation(1, 1))
        with self.assertRaises(FPyException) as cm:
            _evaluate_expr(self.sandbox, expr)
        err = cm.exception.error
        self.assertIsInstance(err, FPySimpleError)
        self.assertIn("not defined", err.message)

    def test_evaluate_and_short_circuit(self):
        # 假设修复：False and 1 -> False, True and 1 -> 1
        source = "False and 1"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertFalse(result)

        source = "True and 42"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 42)

    def test_evaluate_or_short_circuit(self):
        source = "True or 1"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertTrue(result)

        source = "False or 42"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 42)

    def test_evaluate_binop(self):
        source = "2 + 3"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 5)

    def test_evaluate_unaryop(self):
        source = "-5"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, -5)

        source = "not True"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertFalse(result)

    def test_evaluate_compare_chained(self):
        source = "1 < 2 < 3"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertTrue(result)

        source = "1 < 2 > 3"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertFalse(result)

    def test_evaluate_call(self):
        source = "len([1,2])"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 2)

    def test_evaluate_meth_call(self):
        source = "'hello'.upper()"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, "HELLO")

    def test_evaluate_index(self):
        self.sandbox.set_var("lst", [1, 2, 3])
        source = "lst[1]"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 2)

    def test_evaluate_list_with_starred(self):
        # 假设修复：使用 extend
        source = "[1, *[2,3], 4]"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, [1, 2, 3, 4])

    def test_evaluate_dict_with_unpack(self):
        # 假设修复：无变量遮蔽
        source = "{** {'a':1}, 'b':2}"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_evaluate_set_with_starred(self):
        # 假设修复：使用 update
        source = "{1, *[2,3]}"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, {1, 2, 3})

    def test_evaluate_tuple_with_starred(self):
        # 假设修复：使用 extend
        source = "(1, *(2,3))"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, (1, 2, 3))

    def test_evaluate_slice(self):
        self.sandbox.set_var("lst", [1, 2, 3, 4])
        source = "lst[1:3]"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, [2, 3])

    def test_evaluate_binop_error(self):
        source = "1 / 0"
        stmts = parse_fpy(source)
        with self.assertRaises(FPyException) as cm:
            _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        err = cm.exception.error
        self.assertIsInstance(err, FPyExecError)
        self.assertIn("division by zero", err.message)


class TestSandbox(unittest.TestCase):
    """测试沙箱 (DefaultSecureSandbox)"""

    sandbox: DefaultSecureSandbox

    def setUp(self):
        self.sandbox = DefaultSecureSandbox(enable_monitoring=True)

    def test_set_get_var(self):
        self.sandbox.set_var("x", 42)
        self.assertEqual(self.sandbox.get_var("x"), 42)
        self.sandbox.set_var("x", undef)
        self.assertIsInstance(self.sandbox.get_var("x"), Undef)

    def test_list_vars(self):
        self.sandbox.set_var("x", 1)
        self.sandbox.set_var("y", 2)
        vars_list = list(self.sandbox.list_vars())
        self.assertEqual(set(vars_list), {"x", "y"})

    def test_func_call_allowed(self):
        result = self.sandbox.func_call("len", [[1, 2]], SourceLocation(1, 1))
        self.assertEqual(result, 2)

    def test_func_call_not_allowed(self):
        with self.assertRaises(FPyException) as cm:
            self.sandbox.func_call("open", ["file"], SourceLocation(1, 1))
        err = cm.exception.error
        self.assertIsInstance(err, FPySimpleError)
        self.assertIn("not allowed", err.message)

    def test_method_call_allowed(self):
        result = self.sandbox.method_call("hello", "upper", [], SourceLocation(1, 1))
        self.assertEqual(result, "HELLO")

    def test_method_call_private_forbidden(self):
        with self.assertRaises(FPyException) as cm:
            self.sandbox.method_call([], "__len__", [], SourceLocation(1, 1))
        err = cm.exception.error
        self.assertIn("not allowed", err.message)

    def test_recursion_depth_exceeded(self):
        self.sandbox.max_recursion_depth = 1

        def recursive():
            self.sandbox.func_call("recursive", [], SourceLocation(1, 1))

        self.sandbox.add_function("recursive", recursive)
        with self.assertRaises(FPyException) as cm:
            self.sandbox.func_call("recursive", [], SourceLocation(1, 1))
        err = cm.exception.error
        self.assertIn("recursion depth exceeded", err.message)

    def test_display(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            self.sandbox.display(42)
            self.assertIn("[FPy Result] 42", fake_out.getvalue())

    def test_on_error(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            err = FPySimpleError("test", SourceLocation(1, 1))
            self.sandbox.on_error(err)
            self.assertIn("[MONITOR] Error occurred", fake_out.getvalue())

    def test_chaining_compare(self):
        source = "1 < 2 < 3"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 1 < 2 < 3)

        source = "1 < 2 > 3"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 1 < 2 > 3)

        source = "1 < 2 == 3"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 1 < 2 == 3)

        source = "1 < 2 != 3"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 1 < 2 != 3)

        source = "1 < 2 >= 3"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 1 < 2 >= 3)

        source = "1 < 2 <= 3"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 1 < 2 <= 3)

    def test_is_operator(self):
        a = []
        b = []
        self.sandbox.set_var("a", a)  # type: ignore
        self.sandbox.set_var("b", b)  # type: ignore
        self.sandbox.set_var("c", a)  # type: ignore
        source = "a is b"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, a is b)

        source = "a is c"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, a is a)

        source = "a is not b"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, a is not b)

        source = "a is not c"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, a is not a)

    def test_in_operator(self):
        a = [1, 2, 3]
        self.sandbox.set_var("a", a)  # type: ignore
        source = "1 in a"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 1 in a)

        source = "1 not in a"
        stmts = parse_fpy(source)
        result = _evaluate_expr(self.sandbox, stmts[0].value)  # type: ignore
        self.assertEqual(result, 1 not in a)


class TestErrors(unittest.TestCase):
    """测试错误系统 (FPyException, FPyError)"""

    def test_fpy_exception_short_message(self):
        err = FPySimpleError("msg", SourceLocation(1, 1))
        exc = FPyException(err)
        self.assertIn("FPySimpleError", str(exc))

    def test_format_parse_error(self):
        err = FPyParseError("msg", SourceLocation(1, 1, source="x = 1 +\n"))
        buf = io.StringIO()
        FPyException.s_format(buf.write, err)
        output = buf.getvalue()
        self.assertIn("Parse error: msg", output)
        self.assertIn("---> 1 x = 1 +", output)  # 检查 source 提取

    def test_format_simple_error(self):
        err = FPySimpleError("msg", SourceLocation(1, 1))
        buf = io.StringIO()
        FPyException.s_format(buf.write, err)
        self.assertIn("Error: msg", buf.getvalue())

    def test_format_exec_error(self):
        err = FPyExecError("TypeError", "backtrace", "msg", SourceLocation(1, 1))
        buf = io.StringIO()
        FPyException.s_format(buf.write, err)
        output = buf.getvalue()
        self.assertIn("Exec error", output)
        self.assertIn("[Original Python Backtrace]", output)
        self.assertIn("backtrace", output)

    def test_pretty_print(self):
        err = FPySimpleError("msg", SourceLocation(1, 1))
        exc = FPyException(err)
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            exc.pretty_print()
            self.assertIn("Error: msg", fake_out.getvalue())


class TestHighLevel(unittest.TestCase):
    """测试高层API (execute_fpy)"""

    def test_execute_fpy_simple(self):
        source = "len([1,2])"
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            execute_fpy(source)
            self.assertIn("[FPy Result] 2", fake_out.getvalue())

    def test_execute_fpy_with_sandbox(self):
        sandbox = DefaultSecureSandbox()
        source = "x = 1"
        execute_fpy(source, sandbox)
        self.assertEqual(sandbox.get_var("x"), 1)

    def test_execute_fpy_error(self):
        source = "1 / 0"
        with self.assertRaises(FPyException) as cm:
            execute_fpy(source)
        err = cm.exception.error
        self.assertIsInstance(err, FPyExecError)

    def test_special_bin_op(self):
        sandbox = DefaultSecureSandbox()
        data = [1, 2.0, None, True, False, "hello", [1], {1}, {1: 2}, {"1": 2}]
        operators = ["-", "*", "/", "%", "**", "<<", ">>", "&", "|", "^"]
        for d1 in data:
            for d2 in data:
                for op in operators:
                    source = f"{d1!r} {op} {d2!r}"
                    try:
                        execute_fpy("var = " + source, sandbox)
                        self.assertEqual(sandbox.get_var("var"), eval(source))
                    except Exception:
                        with self.assertRaises(FPyException):
                            execute_fpy(source)

    def test_special_unary_op(self):
        sandbox = DefaultSecureSandbox()
        data = [1, 2.0, None, True, False, "hello", [1], {1}, {1: 2}, {"1": 2}]
        operators = ["-", "+", "~", "not"]
        for d in data:
            for op in operators:
                source = f"{op} {d!r}"
                try:
                    execute_fpy("var = " + source, sandbox)
                    self.assertEqual(sandbox.get_var("var"), eval(source))
                except Exception:
                    with self.assertRaises(FPyException):
                        execute_fpy(source)


if __name__ == "__main__":
    unittest.main()

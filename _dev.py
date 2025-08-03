from sandboxed_python import execute_fpy
from sandboxed_python.finite_python import FPyException

execute_fpy("""
print("Hello, World!")
""")

try:
    execute_fpy("""
print([*{1, 2}, 3, *(4, 5), *{5: 1}])

"1" - "2"
    """)
except FPyException as e:
    e.pretty_print()
    
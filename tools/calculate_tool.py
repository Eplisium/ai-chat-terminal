import ast
import operator

def execute(arguments):
    """Execute calculation tool"""
    expression = arguments['expression']
    try:
        def safe_eval(node):
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                return operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = safe_eval(node.operand)
                return operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")
        
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}" 
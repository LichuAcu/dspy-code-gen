import os
from typing import List, Dict
import dspy
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv
import sys

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

lm = dspy.LM('openai/gpt-4o', max_tokens=1000)
dspy.configure(lm=lm)


class CodeSignatureGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("task -> code_signature")

    def forward(self, task: str) -> Dict[str, str]:
        return self.prog(task=task)


class CodeGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(
            "task, code_signature -> code")

    def forward(self, task: str, code_signature: str) -> Dict[str, str]:
        return self.prog(task=task, code_signature=code_signature)


class UnitTestGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(
            "task, code_signature -> test_1, test_2, edge_case_test_1")

    def forward(self, task: str, code_signature: str) -> Dict[str, str]:
        return self.prog(task=task, code_signature=code_signature)


class CodeFixer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(
            "task, old_code, failed_test, error_message -> fixed_code")

    def forward(self, task: str, old_code: str, failed_test: str, error_message: str) -> Dict[str, str]:
        return self.prog(task=task, old_code=old_code, failed_test=failed_test, error_message=error_message)


class CodeGenerationPipeline:
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples
        self.trained_signature = self._compile_module(
            CodeSignatureGenerator, "task")
        self.trained_code_gen = self._compile_module(
            CodeGenerator, "task", "code_signature")
        self.trained_unit_test = self._compile_module(
            UnitTestGenerator, "task", "code_signature")
        self.trained_code_fixer = CodeFixer()

    def _compile_module(self, module_class, *input_fields):
        examples = [dspy.Example(**x).with_inputs(*input_fields)
                    for x in self.examples]
        return BootstrapFewShot().compile(module_class(), trainset=examples)

    def generate(self, task: str) -> None:
        signature_result = self.trained_signature(
            task=f"Write the signature for a Python function doing the following (if the function uses classes, also include the class definition and their methods): {task}")

        code_result = self.trained_code_gen(
            task=f"Write {task} with the provided code signature",
            code_signature=signature_result.code_signature)

        test_result = self.trained_unit_test(
            task=f"Generate unit tests for the following task with the provided code signature: {
                task}",
            code_signature=signature_result.code_signature)

        self._print_results(signature_result, code_result, test_result)
        self._run_code_and_tests(task, code_result.code, test_result)

    def _print_results(self, signature_result, code_result, test_result):
        print(f"\nGenerated code signature:\n{
              signature_result.code_signature}")
        print(f"\nGenerated code:\n{code_result.code}")
        print(f"\nGenerated tests:")
        print(f"test_1: {test_result.test_1}")
        print(f"test_2: {test_result.test_2}")
        print(f"edge_case_test_1: {test_result.edge_case_test_1}")

    def _run_code_and_tests(self, task: str, code: str, test_result):
        print("\nRunning the code...")
        try:
            exec(code)
        except Exception as e:
            print(f"Code failed with error: {str(e)}")
            print("Re-generating the code with the execution feedback...")
            fixed_code = self._fix_code(task, code, "main code", str(e))
            print(f"\nFixed code:\n{fixed_code}")
            self._run_code_and_tests(task, fixed_code, test_result)
            return

        print("Running the tests...")
        for test_name, test_code in [("test_1", test_result.test_1),
                                     ("test_2", test_result.test_2),
                                     ("edge_case_test_1", test_result.edge_case_test_1)]:
            try:
                exec(test_code)
            except Exception as e:
                print(f"Test {test_name} failed with error: {str(e)}")
                print("Re-generating the code with the execution feedback...")
                fixed_code = self._fix_code(task, code, test_code, str(e))
                print(f"\nFixed code:\n{fixed_code}")
                self._run_code_and_tests(task, fixed_code, test_result)
                return

        print("All generated tests passed")

    def _fix_code(self, task: str, old_code: str, failed_test: str, error_message: str) -> str:
        fix_result = self.trained_code_fixer(
            task=task,
            old_code=old_code,
            failed_test=failed_test,
            error_message=error_message
        )

        return fix_result.fixed_code


# Example usage
if __name__ == "__main__":
    examples = [
        {
            "task": "Write a Python function to check if a number is prime with the provided signature",
            "code_signature": "def is_prime(n: int) -> bool:",
            "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
            "test_1": "assert is_prime(2) == True",
            "test_2": "assert is_prime(4) == False",
            "edge_case_test_1": "assert is_prime(1) == False"
        },
        {
            "task": "Create a Python class for a simple calculator with the provided signature",
            "code_signature": "class Calculator { add(a: int, b: int) -> int, subtract(a: int, b: int) -> int, multiply(a: int, b: int) -> int, divide(a: int, b: int) -> int }",
            "code": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    def subtract(self, a, b):\n        return a - b\n    def multiply(self, a, b):\n        return a * b\n    def divide(self, a, b):\n        if b != 0:\n            return a / b\n        else:\n            raise ValueError('Cannot divide by zero')",
            "test_1": "assert Calculator().add(2, 3) == 5",
            "test_2": "assert Calculator().subtract(5, 3) == 2",
            "edge_case_test_1": "assert Calculator().multiply(0, 3) == 0"
        },
        {
            "task": "Write a Python function to reverse a string with the provided signature",
            "code_signature": "def reverse_string(s: str) -> str:",
            "code": "def reverse_string(s):\n    return s[::-1]",
            "test_1": "assert reverse_string('hello') == 'olleh'",
            "test_2": "assert reverse_string('Python') == 'nohtyP'",
            "edge_case_test_1": "assert reverse_string('') == ''"
        },
        {
            "task": "Write a Python function to find the sum of squares of first n natural numbers with the provided signature",
            "code_signature": "def sum_of_squares(n: int) -> int:",
            "code": "def sum_of_squares(n):\n    return sum(i**2 for i in range(1, n+1))",
            "test_1": "assert sum_of_squares(3) == 14",
            "test_2": "assert sum_of_squares(5) == 55",
            "edge_case_test_1": "assert sum_of_squares(0) == 0"
        }
    ]

    pipeline = CodeGenerationPipeline(examples)

    default_task = "A Python function to get the nth Fibonacci number"
    if len(sys.argv) > 2 and sys.argv[1] == "--task":
        task = sys.argv[2]
    else:
        task = default_task

    pipeline.generate(task)

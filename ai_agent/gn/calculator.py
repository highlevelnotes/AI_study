"""
간단한 계산기 프로그램
덧셈, 뺄셈 기능을 제공합니다.
"""


def add(a, b):
    """두 수의 덧셈"""
    return a + b


def subtract(a, b):
    """두 수의 뺄셈"""
    return a - b


def calculator():
    """계산기 메인 함수"""
    print("=" * 50)
    print("계산기 프로그램")
    print("=" * 50)
    print("1. 덧셈 (+)")
    print("2. 뺄셈 (-)")
    print("0. 종료")
    print("=" * 50)
    
    while True:
        try:
            choice = input("\n연산을 선택하세요 (0-2): ").strip()
            
            if choice == '0':
                print("계산기를 종료합니다.")
                break
            
            if choice not in ['1', '2']:
                print("잘못된 선택입니다. 0-2 사이의 숫자를 입력하세요.")
                continue
            
            try:
                num1 = float(input("첫 번째 숫자를 입력하세요: "))
                num2 = float(input("두 번째 숫자를 입력하세요: "))
            except ValueError:
                print("올바른 숫자를 입력하세요.")
                continue
            
            result = None
            operator = ""
            
            if choice == '1':
                result = add(num1, num2)
                operator = "+"
            elif choice == '2':
                result = subtract(num1, num2)
                operator = "-"
            
            print(f"\n결과: {num1} {operator} {num2} = {result}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n계산기를 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    calculator()


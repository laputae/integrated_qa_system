import argparse

def main():
    parser = argparse.ArgumentParser(description='一个简单的示例程序')

    parser.add_argument('input', help='输入文件路径')
    parser.add_argument('--output', required=False, default='output.txt', help='输出文件路径')
    parser.add_argument('--verbose', action='store_true', help='是否打印详细信息')
    parser.add_argument('--times', type=int, default=1, help='重复处理次数')

    args = parser.parse_args()

    if args.verbose:
        print(f"正在处理 {args.input} -> {args.output}，重复 {args.times} 次")

    for i in range(args.times):
        print(f"处理中... {i+1}/{args.times}")

if __name__ == '__main__':
    main()
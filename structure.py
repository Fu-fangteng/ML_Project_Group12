import os

def print_directory_tree(start_path, indent='', is_last=True):
    """递归打印文件夹结构"""
    basename = os.path.basename(start_path)
    prefix = '└── ' if is_last else '├── '
    print(indent + prefix + basename)
    indent += '    ' if is_last else '│   '
    
    if os.path.isdir(start_path):
        try:
            items = sorted(os.listdir(start_path))
        except PermissionError:
            print(indent + '└── [权限被拒绝]')
            return

        for i, item in enumerate(items):
            full_path = os.path.join(start_path, item)
            is_last_item = (i == len(items) - 1)
            print_directory_tree(full_path, indent, is_last_item)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='打印指定目录的文件结构')
    parser.add_argument('path', help='目标文件夹路径')
    args = parser.parse_args()

    if os.path.exists(args.path):
        print_directory_tree(args.path)
    else:
        print("路径不存在，请提供有效路径。")

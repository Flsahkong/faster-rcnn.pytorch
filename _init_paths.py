import os.path as osp
import sys

# 这里是添加包的路径的,具体的sys.path解析需要看笔记
# 在这里添加的path在进程结束后就会消失
# insert(0,path)表示添加到开头,这样子保证运行的时候首先找到本项目的path运行
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

coco_path = osp.join(this_dir, 'data', 'coco', 'PythonAPI')
add_path(coco_path)

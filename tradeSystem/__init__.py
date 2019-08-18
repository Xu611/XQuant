# encoding: UTF-8
# os.path.abspath(os.path.join(os.getcwd(), "../..")) # 返回上上级目录

import os,sys

parentdir = os.path.dirname(__file__)  # 返回当前的工作目录
if parentdir not in sys.path:
    sys.path.insert(0,parentdir)

topdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
targetdir = topdir+'\\toolbox'
if targetdir not in sys.path:
    sys.path.insert(0,targetdir)



__version__ = '1.0.0'
__author__  = 'xujw'
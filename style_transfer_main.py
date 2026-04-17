#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI艺术创作工作室主入口
启动图形用户界面
"""

import os
import sys
import tkinter as tk

# 导入UI模块
from style_transfer_ui import main as ui_main

if __name__ == "__main__":
    print("🎨 启动AI艺术创作工作室...")
    ui_main()
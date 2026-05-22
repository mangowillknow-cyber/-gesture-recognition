# Gesture Recognition

基于 MediaPipe 的手势识别桌面应用。

## 功能

- 手势识别：数字 1-10（双手支持 6-10）、握拳、点赞、OK
- 摄像头切换：下拉菜单选择 + 刷新按钮
- 支持 Iriun Webcam（手机摄像头）
- 中文界面
- 实时手势可视化

## 技术栈

- Python 3.10+
- OpenCV
- MediaPipe Tasks API
- PyQt5

## 运行

```bash
pip install -r requirements.txt
python gesture_recognition.py
```

## 项目结构

```
gesture_recognition.py   # 主程序（单文件）
hand_landmarker.task     # MediaPipe 手部模型
icon.png                 # 应用图标
requirements.txt         # 依赖
tests/                   # 单元测试
```

## 开发

本项目由 [Claude](https://claude.ai) 协助开发，包括需求分析、架构设计、代码实现和测试。

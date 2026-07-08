# 本地模型识别服务

这个目录提供本地 HTTP 服务，用于让网页端把「图例裁剪」和「整张网格截图」发到本机进行批量识别。

默认只依赖本地 OCR，不需要 OpenAI API Key，也不会调用外部 AI 接口。若配置了本地/兼容视觉大模型，可额外使用「AI识别」读取图例。

## 安装依赖

在项目目录下运行：

```bash
python3 -m pip install -r local_ai_server/requirements.txt
```

## 启动

推荐在项目根目录直接运行：

```bash
./start.command
```

它会同时启动网页服务和本地识别服务。识别服务会固定使用 `/usr/bin/python3`，避免误用缺少 `rapidocr` 的 Python 环境。

如果只想单独启动本地识别服务：

```bash
/usr/bin/python3 local_ai_server/server.py
```

默认监听：

- http://127.0.0.1:5055
- 健康检查：GET /health

网页会固定请求 `http://127.0.0.1:5055`。如果页面提示连接失败，请先确认本服务已启动。

## 可选：智谱 GLM 识别

Step1 的「GLM」和 Step2 的「GLM识别」走智谱 OpenAI 兼容接口。它和本地 OCR 是两条独立路径，方便对比识别效果。

创建 `local_ai_server/vlm.env`：

```bash
export PINDOU_GLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export PINDOU_GLM_MODEL="glm-4v-flash"
export PINDOU_GLM_API_KEY="你的智谱 API Key"
```

保存后重新运行：

```bash
./start.command
```

如果模型名称不同，只改 `PINDOU_GLM_MODEL`。`local_ai_server/vlm.env` 只保存在本机，已被 `.gitignore` 排除，不要提交到 GitHub。

## 接口

### POST /legend

请求（JSON）：

```json
{
  "image": "data:image/jpeg;base64,...",
  "validCodes": ["H1", "H2"]
}
```

响应（JSON）：

```json
{
  "items": [
    { "code": "H2", "count": 225, "conf": 0.9 }
  ]
}
```

### POST /legend-glm

请求同 `/legend`。响应：

```json
{
  "source": "glm",
  "items": [
    { "code": "H2", "count": 112, "conf": 0.98 }
  ]
}
```

### POST /grid

请求（JSON）：

```json
{
  "image": "data:image/jpeg;base64,...",
  "rows": 40,
  "cols": 40,
  "allowedCodes": ["H1", "H2"]
}
```

响应（JSON）：

```json
{
  "cells": [
    { "r": 0, "c": 0, "code": "H2", "conf": 0.9 }
  ]
}
```

## 说明

- 当前 `server.py` 使用 RapidOCR（ONNXRuntime）在本地进行 OCR。
- `/legend` 用本地 OCR 识别图例区域中的色号和数量。
- `/legend-glm` 用智谱 GLM 识别图例区域，适合水印、网格干扰较重的情况。
- `/grid` 用本地 OCR 按行条带识别网格中的文字，并映射回格子坐标。
- 如果你想换成 PaddleOCR/自定义模型，可以在 `server.py` 中替换 `_run_ocr()` 的实现。

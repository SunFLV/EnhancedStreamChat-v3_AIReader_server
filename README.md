# 🎙 AI 语音朗读 Web UI

一个基于 GPT-SoVITS TTS 服务的现代化网页界面，支持**弹幕朗读**与**长文有声书合成**。

纯前端单文件应用（HTML + CSS + JS），通过 REST API 对接任意 GPT-SoVITS 语音合成后端。

---

## ✨ 功能特性

### 🗨 弹幕朗读
- 输入文本，选择声音，一键合成并播放
- 支持**中英混合**输入
- 可选「说话人」前缀，自动拼接「xxx说」并缓存（同一观众多条弹幕只推理一次）

### 📖 长文合成（有声小说）
- 粘贴整章 / 整本文本（最多 10 万字），自动按句分段
- 逐句合成，**实时进度条**显示 `正在合成 12/35 句（34%）…`
- 支持**断点续传**：合成中途崩溃，重启后从上次完成处继续
- 任务结果**磁盘持久化 12 小时**，凭任务ID + 密钥随时取回

### 🔁 进度恢复
- 每个长文任务返回「任务ID + 密钥」
- 断网 / 刷新页面后，凭 ID + 密钥恢复进度并下载音频
- 12 小时内有效，过期自动清理

### 🎚 其他
- 语速调节（0.5x - 2.0x）
- 多个音色预设下拉切换
- 深色渐变现代界面

---

## 🖥 界面

两个标签页切换：

| 标签页 | 用途 |
|--------|------|
| **弹幕朗读** | 短文本 + 说话人前缀，实时合成 |
| **长文合成** | 整章文本，进度条 + 任务恢复 |

---

## 🔌 对接的 API

前端通过以下 REST 接口与 TTS 后端通信（**需后端提供**）：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/voices` | 获取可用声音列表 |
| `POST` | `/api/tts` | 单句合成 → 返回 `audio/wav` |
| `POST` | `/api/tts/long` | 提交长文任务 → 返回 `{task_id, key}` |
| `GET` | `/api/tts/progress/{task_id}?key=xxx` | 查询合成进度 |
| `GET` | `/api/tts/result/{task_id}?key=xxx` | 取回完整音频 |

### 请求示例

```json
// POST /api/tts  （单句）
{
  "speaker": "主播名",       // 可选，为空则只读文本
  "text": "欢迎来到直播间",
  "voice": "default",
  "speed": 1.0
}

// POST /api/tts/long  （长文）
{
  "text": "整章小说文本……",
  "voice": "default",
  "speed": 1.0
}
```

### 长文任务流程

```
POST /api/tts/long → {task_id, key}
    ↓ 轮询
GET /api/tts/progress/{task_id}?key=xxx → {status, done, total}
    ↓ 完成后
GET /api/tts/result/{task_id}?key=xxx → 完整 WAV 音频
```

---

## 🚀 部署

### 方式一：嵌入后端（推荐）

前端内嵌于 TTS 代理服务中，启动后直接访问根路径：

```bash
# 启动 TTS 后端 + 代理
./start_all.sh   # 或对应平台的启动脚本

# 打开浏览器
http://localhost:8000
```

### 方式二：独立静态部署

将 `index.html` 部署到任意静态服务器（Nginx / GitHub Pages / CDN），并配置前端中的 `API_BASE` 指向后端地址：

```js
// 在 index.html 中配置
const API_BASE = 'https://your-tts-server.example.com';
```

> 注意：若前后端不同源，需后端开启 CORS。

---

## 🛠 技术栈

- **纯原生** HTML + CSS + JavaScript（无框架、无构建）
- REST API 对接 GPT-SoVITS 语音合成
- Web Audio 播放

---

## 📦 目录结构

```
web/
├── index.html      # 单文件应用（全部逻辑）
└── README.md       # 本文件
```

---

## 📄 许可

MIT License

---

## 🙏 致谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 强大的开源 TTS 引擎

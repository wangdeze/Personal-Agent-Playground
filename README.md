# Personal-Agent-Playground

## 设计目标
超级轻量化的脚本驱动 Agent Prompt 本地测试框架。仅用于测试 Prompt，不增加与该目标无关的功能或依赖。

## 基本结构
```text
agent-playground/
├── prompt.md       # System Prompt
├── config.yaml     # API / model / 参数
├── chat.py         # 主程序
├── test_chat.py    # 标准库单元测试
└── history/        # 自动保存的对话与 API 调用历史
```

## 必要约束
1. 轻量优先，只使用 Python 标准库。
2. 通过 OpenAI Chat Completions 兼容 API 调用互联网 LLM。
3. 默认每个对话最多 10 轮用户输入，可在配置中调整。
4. 自动保存并加载最近会话，每轮按 `user_input`、`api_call`、`assistant_response` 连续归档。
5. 不提供本地工具接口，仅调试对话功能。
6. 支持 `reset`、`clean`、`reload`、`save`、`quit` 等 CLI 命令。

## 使用

需要 Python 3.10 或更新版本。将 API Key 放入项目根目录的 `.env`（该文件已加入 `.gitignore`），编辑 `prompt.md` 和 `config.yaml` 后运行：

```sh
OPENROUTER_API_KEY="your-api-key"
python3 chat.py
```

配置中的 `base_url` 应为 API 版本根路径（例如 `https://api.openai.com/v1`）；程序会在其后追加 `/chat/completions`。`api_key_env` 指定 `.env` 中的密钥变量名；已有的 shell 环境变量优先于 `.env`。配置文件支持简单的扁平 `key: value` 标量，不需要 YAML 第三方依赖。

在交互提示符输入 `/help` 查看命令。`/reset` 保留当前存档并开始一个新的 session 文件；`/clean` 删除所有存档并开始新会话；`/reload` 重新读取配置和 Prompt。会话存档位于 `history/`，请求日志不包含 API Key。旧格式存档会在加载时自动迁移。

运行测试：

```sh
python3 -m unittest -v
```
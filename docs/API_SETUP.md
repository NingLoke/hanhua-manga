# API 翻译设置与 API Key 申请教程

**硬件性能不足时，强烈推荐使用 API 翻译。** 云端服务运行翻译大模型，你的电脑继续负责检测、OCR、修复和排版；不需要下载 GGUF 翻译模型或 llama.cpp。API 不会免除所有本地计算，但能免去运行翻译大模型的开销。

以下官网文档在 2026-10-31 核对。服务商可能修改界面、模型名称、地区限制、价格或配额，最终以你的账号控制台为准。

## 先在软件里这样设置

1. 打开「设置 → 翻译」。
2. **关闭「本地大模型」**。4.0.5 默认优先使用本地模型，仅填入 Key 不会改变这个优先级。
3. 开启「API 服务（填 Key）」，选择对应服务商。
4. 粘贴 API Key。使用预设服务商时，API 地址留空可使用内置地址。
5. 点「获取模型列表」并选模型；若平台不返回模型列表，手动填写该平台提供的完整模型 ID。
6. 保存，点「测试连接」，然后先汉化少量页面。连接测试会发送请求，可能计费。
7. 想确保只使用选定 API，可关闭「免费在线翻译（兜底）」。

API Key 是平台发给你的调用凭证，不是网页登录密码，也不是聊天网页里的分享链接。请填写文本对话模型，不要选图片生成、语音、向量或重排模型。模型需支持 OpenAI 兼容 Chat Completions 以及 JSON Schema / JSON 对象输出。

## DeepSeek

- [官方平台 / 申请 Key](https://platform.deepseek.com/)
- [API Keys 页面](https://platform.deepseek.com/api_keys)
- [官方 API 入门](https://api-docs.deepseek.com/)

注册或登录官方平台，进入「API Keys」，创建一个用于本软件的 Key 并复制。按平台要求检查余额和模型权限。在软件中选择 **DeepSeek**、粘贴 Key，再获取并选择当前可用的模型。

预设 API 地址为 `https://api.deepseek.com/v1`。4.0.5 历史默认模型名是 `deepseek-chat`；厂商模型会更新，**不要把旧默认值当成当前可用保证**，优先从模型列表或官方文档填写有效 ID。

## 硅基流动 SiliconFlow

- [官方控制台](https://cloud.siliconflow.cn/)
- [API 密钥管理](https://cloud.siliconflow.cn/account/ak)
- [官方快速上手](https://docs.siliconflow.cn/docs/userguide/quickstart)

注册或登录控制台，打开「API 密钥」，点「新建 API 密钥」后复制。查看平台模型列表，确认目标文本对话模型可用。在软件中选择 **硅基流动**，填写 Key，获取模型列表并选择完整模型 ID。

预设 API 地址为 `https://api.siliconflow.cn/v1`。模型 ID 可能包含组织前缀，不能只填写显示用的模型简称。

## Google Gemini

- [Google AI Studio / API Keys](https://aistudio.google.com/apikey)
- [官方 Key 说明](https://ai.google.dev/gemini-api/docs/api-key)
- [官方 OpenAI 兼容接口说明](https://ai.google.dev/gemini-api/docs/openai)

用 Google 账号登录 AI Studio，在 API Keys 页面创建 / 选择项目并创建 API Key，按页面要求完成项目配置。在软件中选 **Google Gemini**，填写 Key，再选择该项目允许调用的文本对话模型。地区、项目权限、免费额度与结算要求以 Google 控制台为准。

本软件使用兼容接口，地址为 `https://generativelanguage.googleapis.com/v1beta/openai`。不要填写 Gemini 网页聊天地址或原生 `generateContent` 请求地址。

## OpenAI

- [OpenAI Platform](https://platform.openai.com/)
- [项目 API Keys](https://platform.openai.com/api-keys)
- [OpenAI Docs：官方快速上手](https://developers.openai.com/api/docs/quickstart)

登录 Platform，选择你的项目，在 API Keys 页面创建一个应用 API Key，复制到软件的 **OpenAI** 预设中。按账号要求配置 API 结算或额度，再选择项目可访问、支持 Chat Completions 与所需 JSON 输出的文本模型。

预设 API 地址为 `https://api.openai.com/v1`。软件不需要你创建管理整个组织的 Admin Key，也不需要把 Key 发给项目作者。

## OpenRouter

- [官方网站](https://openrouter.ai/)
- [Key 管理](https://openrouter.ai/settings/keys)
- [官方快速上手](https://openrouter.ai/docs/quickstart)

登录后在 Keys 页面创建 Key，按需设置额度限制。在软件中选择 **OpenRouter**，填写 Key，并填写 / 选择完整模型 ID。不同模型背后的上游供应商、收费与兼容能力不同，先用少量页面测试。

预设 API 地址为 `https://openrouter.ai/api/v1`。

## 自定义服务 / Ollama / LM Studio

选择「自定义（OpenAI 兼容）」，填写服务商的 Base URL、Key 和模型 ID。Base URL 到 `/v1` 等根路径为止，**不要再附加 `/chat/completions`**，软件会自动拼接。

本机运行的兼容服务可以使用 `http://127.0.0.1:11434/v1` 等实际地址；服务本身仍需你启动并加载模型，本机模式不会替低配电脑省掉模型计算。

## 常见问题

- **Key 有效但翻译仍很慢**：确认关闭了本地大模型；查看耗时发生在 OCR / 修复还是翻译步骤，API 只能减少翻译模型的本地开销。
- **401 / 403**：核对 Key、服务商、项目和模型访问权限。Key 创建成功不等于模型授权成功。
- **404 / 模型不存在**：获取当前模型列表，手动更新模型 ID，检查 Base URL 是否填对。
- **429**：配额、速率限制或余额问题；降低同时翻译请求数并查看平台控制台。
- **400 / JSON 输出不兼容**：换支持 JSON 输出的文本对话模型；不是平台上所有模型都适合这个翻译客户端。
- **连接失败**：检查网络和地区可达性。不要把其他平台的 Key 填到不信任的第三方中转站。

## Key 与台词会去哪里

Key 保存在本机设置中，4.0.5 没有系统密钥库加密；任务设置快照也可能包含 Key。不要上传设置文件、任务目录、日志、缓存或带 Key 的截图。

调用 API 时，Key 会用于向你填写的服务地址认证，台词、部分上下文和术语表会发送给该服务。不会因使用 API 自动把整本漫画打包上传；图片处理步骤仍在本机进行。

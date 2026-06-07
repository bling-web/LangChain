# DashScope Compatible Mode

This project uses DashScope through the OpenAI-compatible endpoint, so it does not need `langchain-dashscope`.

Set the API key before running:

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"
```

Run the example:

```powershell
$env:UV_CACHE_DIR='E:\pythonPrograming\bigModelTest\.uv-cache'
uv run python main.py
```

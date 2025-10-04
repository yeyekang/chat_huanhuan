from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id='LLM-Research/Meta-Llama-3.1-8B-Instruct',
    cache_dir='autodl-tmp/chat-huanhuan/model',
    revision='master'
)

print("✅ 下载完成，目录：", model_dir)

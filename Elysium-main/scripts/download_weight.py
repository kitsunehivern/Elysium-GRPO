from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sty-yyj/elysium_7b",
    local_dir=f"./checkpoints/elysium_7b",
    local_dir_use_symlinks=False
)
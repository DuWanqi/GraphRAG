#!/bin/bash
set -e

# 当前目录：GraphRAG
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 上一级：RAG
RAG_ROOT="$(dirname "$PROJECT_ROOT")"

# 默认 Ollama 根目录
OLLAMA_ROOT="$RAG_ROOT/Ollama"

# 默认路径
OLLAMA_BIN="$OLLAMA_ROOT/bin/ollama"
OLLAMA_LOG="$OLLAMA_ROOT/ollama.log"

# 参数
MODE="gradio"
PORT=8000
RESTART_OLLAMA=0

while [[ $# -gt 0 ]]; do
    case $1 in
        api) MODE="api"; shift ;;
        --port) PORT="$2"; shift 2 ;;
        --ollama_path)
            OLLAMA_ROOT="$2"
            shift 2
            ;;
        --restart_ollama)
            RESTART_OLLAMA=1
            shift
            ;;
        *) shift ;;
    esac
done

# ⚠️ 关键：根据传入 ROOT 重新计算路径
OLLAMA_BIN="$OLLAMA_ROOT/bin/ollama"
OLLAMA_LOG="$OLLAMA_ROOT/ollama.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动历史回忆录生成平台"
echo "模式: $MODE | 端口: $PORT"
echo "Ollama ROOT: $OLLAMA_ROOT"
echo "Ollama BIN:  $OLLAMA_BIN"
echo "Ollama LOG:  $OLLAMA_LOG"
echo "Restart Ollama: $RESTART_OLLAMA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# === 1. 启动 Ollama ===
echo "[1/2] 检查 Ollama..."

OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://localhost:11434}"

# 关键：让 Ollama 使用指定目录下的模型，而不是默认 ~/.ollama
export OLLAMA_MODELS="${OLLAMA_MODELS:-$OLLAMA_ROOT/models}"
mkdir -p "$OLLAMA_MODELS"

if curl -s "$OLLAMA_API_BASE/api/tags" > /dev/null 2>&1; then
    if [ "$RESTART_OLLAMA" = "1" ]; then
        echo "  → 检测到 Ollama 已运行，按需重启以应用 OLLAMA_MODELS..."
        # 尝试停止旧的 ollama serve（避免仍使用旧模型目录）
        pkill -f "ollama serve" > /dev/null 2>&1 || true
        sleep 1
    else
        echo "  ✓ Ollama 已运行（如需应用 --ollama_path 的模型目录，请加 --restart_ollama）"
    fi
fi

if ! curl -s "$OLLAMA_API_BASE/api/tags" > /dev/null 2>&1; then
    echo "  → 启动 Ollama..."

    if [ ! -x "$OLLAMA_BIN" ]; then
        echo "  ✗ 未找到 Ollama 可执行文件: $OLLAMA_BIN"
        exit 1
    fi

    # 确保日志目录存在
    mkdir -p "$OLLAMA_ROOT"

    nohup "$OLLAMA_BIN" serve > "$OLLAMA_LOG" 2>&1 &

    for i in {1..10}; do
        if curl -s "$OLLAMA_API_BASE/api/tags" > /dev/null 2>&1; then
            echo "  ✓ Ollama 启动成功"
            break
        fi
        sleep 1
        if [ $i -eq 10 ]; then
            echo "  ✗ Ollama 启动失败"
            exit 1
        fi
    done
fi

# === 2. 启动应用 ===
echo "[2/2] 启动应用..."

cd "$PROJECT_ROOT"

if [ "$MODE" = "api" ]; then
    python3 run_web.py api --host 0.0.0.0 --port "$PORT"
else
    python3 run_web.py gradio --host 0.0.0.0 --port "$PORT"
fi
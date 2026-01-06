#!/bin/bash
# 营销素材工具 - 统一启动脚本
# 同时启动后端 API 服务和前端开发服务器

set -e

echo "🚀 启动营销素材工具..."
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 检查并安装前端依赖
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "📦 检测到未安装前端依赖，正在安装..."
    cd "$SCRIPT_DIR/frontend"
    npm install
    cd "$SCRIPT_DIR"
fi

# 启动后端 (端口 8080)
echo "📡 启动后端 API 服务 (端口 8080)..."
cd "$SCRIPT_DIR/backend"
# 使用 main.py 启动，它会调用 uvicorn
uv run main.py &
BACKEND_PID=$!

# 等待后端启动
sleep 2

# 启动前端 (端口 5173)
echo "🎨 启动前端开发服务器 (端口 5173)..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ 服务已启动!"
echo "   前端: http://localhost:5173"
echo "   后端: http://localhost:8080"
echo "   API 文档: http://localhost:8080/docs"
echo ""
echo "按 Ctrl+C 停止所有服务..."

# 捕获终止信号，停止所有进程
cleanup() {
    echo ""
    echo "🛑 正在停止服务..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# 等待进程
wait

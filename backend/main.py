import uvicorn

if __name__ == "__main__":
    # 启动后端服务
    # reload=True 在开发模式下很有用，但在生产环境应关闭
    uvicorn.run("api.server:app", host="0.0.0.0", port=8080, reload=True)

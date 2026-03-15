"""Entry point for: python -m backtest_server"""
import uvicorn
from backtest_server.server import app, MODEL_METADATA

print(f"""
╔══════════════════════════════════════════════╗
║       Qlib 回测实验室 API Server             ║
╠══════════════════════════════════════════════╣
║  http://localhost:8001                       ║
║  模型: {len(MODEL_METADATA)} 个                            ║
╚══════════════════════════════════════════════╝
""")
uvicorn.run("backtest_server.server:app", host="0.0.0.0", port=8001, reload=True)

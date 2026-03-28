# === API 路由端点 ===
@app.get("/api/v1/health", response_model=HealthStatusResponse, tags=["System"])
async def check_health():
    """系统健康度检查接口"""
    return HealthStatusResponse(
        status="Running",
        engine_ready=engine.use_omc,
        ai_model_loaded=engine.ai_heat_model is not None,
        server_time=time.time()
    )

@app.post("/api/v1/simulate/predict", tags=["Simulation"])
async def run_twin_prediction(req: SimulationRequest):
    """
    接收外部指令，执行单次数字孪生预测
    """
    try:
        # 调用核心引擎计算
        df, kpis = engine.run_profile(
            time_total=req.duration_s,
            pack_current_a=req.pack_current,
            env_temp_c=req.env_temp,
            init_soc=req.init_soc,
            init_soh=req.init_soh,
            cooling_type=req.cooling_mode
        )
        
        # 封装返回结果
        return {
            "code": 200,
            "message": "数字孪生解算成功",
            "data": {
                "kpis": kpis,
                "time_series_summary": {
                    "max_temp": float(df['Max_Temp'].max()),
                    "final_soc": float(df['SOC'].iloc[-1]),
                    "data_points": len(df)
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"孪生引擎解算崩溃: {str(e)}")

@app.get("/api/v1/aging/estimate", tags=["Aging"])
async def estimate_rul(current_soh: float, avg_temp: float, avg_c_rate: float):
    """
    快速评估剩余循环寿命 (RUL)
    """
    if current_soh < 80.0:
        return {"warning": "电池已达到退役标准 (EOL)", "remaining_cycles": 0}
        
    # 模拟快速衰减计算
    step_loss = AgingModel.calculate_step_loss(avg_temp, avg_c_rate * 100, 50, 3600, 100)
    est_cycles = int((current_soh - 80.0) / (step_loss * 100)) if step_loss > 0 else 9999
    
    return {
        "current_soh": current_soh,
        "estimated_remaining_cycles": est_cycles,
        "advice": "建议降低高倍率充放电频率" if avg_c_rate > 1.0 else "工况良好"
    }

if __name__ == "__main__":
    # 使用 uvicorn 启动 ASGI 服务器
    uvicorn.run(app,host='0.0.0.0',port=8000,log_level="info")
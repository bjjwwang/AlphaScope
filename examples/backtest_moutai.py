"""
贵州茅台（SH600519）简单选股策略回测示例

流程：
1. 用 CSI300 成分股 + Alpha158 因子 训练 LightGBM 模型
2. 模型预测所有股票的未来收益
3. 用 TopkDropoutStrategy 从 CSI300 中选股，回测
4. 同时单独展示茅台的预测信号

这是一个端到端的示例：数据 → 特征 → 模型 → 预测 → 回测 → 分析
"""

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import pandas as pd

# ============================================================
# 1. 初始化 Qlib
# ============================================================
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)
print("Qlib 初始化完成")

# ============================================================
# 2. 配置模型（LightGBM）
# ============================================================
model_config = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    },
}

# ============================================================
# 3. 配置数据集
#    - 用 CSI300 成分股训练，这样模型能学到市场整体规律
#    - Alpha158: 158 个经典量价因子
#    - 标签(label): 未来两天的收益率 Ref($close,-2)/Ref($close,-1)-1
# ============================================================
dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2008-01-01",
                "end_time": "2020-08-01",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2014-12-31",
                "instruments": "csi300",
            },
        },
        "segments": {
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-08-01"),
        },
    },
}

# ============================================================
# 4. 配置回测策略
#    - TopkDropoutStrategy: 每天选 top 50 股票持仓
#    - 每天换 5 只（n_drop=5）
#    - 初始资金 1 亿
# ============================================================
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": "<PRED>",
            "topk": 50,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

# ============================================================
# 5. 运行完整实验
# ============================================================
print("=" * 60)
print("正在初始化模型和数据集...")
model = init_instance_by_config(model_config)
dataset = init_instance_by_config(dataset_config)

print("=" * 60)
print("开始实验...")

with R.start(experiment_name="moutai_backtest"):
    # 训练模型
    print("[1/4] 训练 LightGBM 模型（CSI300, Alpha158 因子）...")
    model.fit(dataset)
    R.save_objects(**{"params.pkl": model})

    recorder = R.get_recorder()

    # 生成预测信号
    print("[2/4] 生成预测信号...")
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # 信号分析（IC, 收益等）
    print("[3/4] 信号质量分析...")
    sar = SigAnaRecord(recorder)
    sar.generate()

    # 组合回测
    print("[4/4] 运行组合回测...")
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()

    # ========================================================
    # 6. 提取茅台（SH600519）的预测信号
    # ========================================================
    print("=" * 60)
    print("提取贵州茅台（SH600519）的预测信号...")

    pred_df = recorder.load_object("pred.pkl")

    # pred_df 是 MultiIndex: (datetime, instrument)
    if "SH600519" in pred_df.index.get_level_values(1):
        moutai_pred = pred_df.xs("SH600519", level=1)
        print("\n贵州茅台 预测信号（测试期 2017-2020）:")
        print(moutai_pred.describe())
        print("\n最近 10 天的预测信号:")
        print(moutai_pred.tail(10))

        # 看看茅台在每天所有股票中的排名
        ranks = pred_df.groupby(level=0).rank(ascending=False, pct=True)
        if "SH600519" in ranks.index.get_level_values(1):
            moutai_rank = ranks.xs("SH600519", level=1)
            print("\n茅台每日排名百分位（越小越好，<0.17 意味着进入 top50/300）:")
            print(moutai_rank.describe())
            top50_days = (moutai_rank < 50.0 / 300).sum().iloc[0] if len(moutai_rank.columns) > 0 else 0
            total_days = len(moutai_rank)
            print(f"\n茅台进入 Top50 选股池的天数: {top50_days} / {total_days} ({100*top50_days/total_days:.1f}%)")
    else:
        print("警告: 未找到 SH600519 的预测数据，可能不在 CSI300 成分股中")

print("\n" + "=" * 60)
print("回测完成！结果已保存到 MLflow")
print("可以运行 'mlflow ui' 查看详细实验结果")

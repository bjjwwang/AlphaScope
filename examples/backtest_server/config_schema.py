"""
Pydantic models for backtest request/response validation.
"""
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, field_validator, model_validator

# Known valid model classes
VALID_MODEL_CLASSES = {
    "LGBModel", "XGBModel", "CatBoostModel", "LinearModel", "DEnsembleModel",
    "LSTM_flat", "LSTM_ts", "GRU_flat", "GRU_ts", "ALSTM_flat", "ALSTM_ts",
    "ADARNN", "Transformer_flat", "Transformer_ts", "LocalFormer_ts",
    "GATs_flat", "GATs_ts", "TRAModel",
    "TCN_flat", "TCN_ts", "TabnetModel", "HIST", "SFM", "DNNModelPytorch",
}

VALID_DATA_SOURCES = {"db", "yfinance", "baostock"}
VALID_KLINES = {"daily", "weekly", "60min", "30min", "15min"}
VALID_TRAINING_POOLS = {"all", "sp500", "volatile", "same_sector", "target_only"}
VALID_FEATURE_SETS = {"Alpha158", "Alpha360", "Alpha158_20", "custom"}
VALID_DATASET_CLASSES = {"DatasetH", "TSDatasetH"}
VALID_STRATEGY_CLASSES = {"TopkDropoutStrategy", "SignalThreshold"}


class ModelConfig(BaseModel):
    model_class: str
    module_path: str
    kwargs: Dict[str, Any] = {}

    @field_validator("model_class")
    @classmethod
    def validate_model_class(cls, v):
        if v not in VALID_MODEL_CLASSES:
            raise ValueError(f"Unknown model: {v}. Must be one of {VALID_MODEL_CLASSES}")
        return v


class SegmentsConfig(BaseModel):
    train: Tuple[str, str]
    valid: Tuple[str, str]
    test: Tuple[str, str]

    @model_validator(mode="after")
    def validate_no_overlap(self):
        if self.train[1] >= self.valid[0]:
            raise ValueError(f"Train end ({self.train[1]}) must be before valid start ({self.valid[0]})")
        if self.valid[1] >= self.test[0]:
            raise ValueError(f"Valid end ({self.valid[1]}) must be before test start ({self.test[0]})")
        return self


class DatasetConfig(BaseModel):
    dataset_class: str
    handler: str
    feature_set: str = "Alpha158"
    custom_features: Optional[List[str]] = None
    step_len: Optional[int] = None
    segments: SegmentsConfig

    @field_validator("dataset_class")
    @classmethod
    def validate_dataset_class(cls, v):
        if v not in VALID_DATASET_CLASSES:
            raise ValueError(f"Invalid dataset class: {v}")
        return v

    @field_validator("feature_set")
    @classmethod
    def validate_feature_set(cls, v):
        if v not in VALID_FEATURE_SETS:
            raise ValueError(f"Invalid feature set: {v}")
        return v


class StrategyConfig(BaseModel):
    strategy_class: str
    topk: Optional[int] = None
    n_drop: Optional[int] = None
    buy_threshold: Optional[float] = None
    capital: float = 100000

    @field_validator("strategy_class")
    @classmethod
    def validate_strategy_class(cls, v):
        if v not in VALID_STRATEGY_CLASSES:
            raise ValueError(f"Invalid strategy: {v}")
        return v

    @field_validator("capital")
    @classmethod
    def validate_capital(cls, v):
        if v <= 0:
            raise ValueError("Capital must be positive")
        return v

    @model_validator(mode="after")
    def validate_strategy_params(self):
        if self.strategy_class == "TopkDropoutStrategy" and self.topk is None:
            raise ValueError("topk is required for TopkDropoutStrategy")
        if self.strategy_class == "SignalThreshold" and self.buy_threshold is None:
            raise ValueError("buy_threshold is required for SignalThreshold")
        return self


class BacktestRequest(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig
    data_source: str
    kline: str = "daily"
    target: List[str]
    training_pool: str = "target_only"
    label: str
    strategy: StrategyConfig

    @field_validator("data_source")
    @classmethod
    def validate_data_source(cls, v):
        if v not in VALID_DATA_SOURCES:
            raise ValueError(f"Invalid data source: {v}")
        return v

    @field_validator("kline")
    @classmethod
    def validate_kline(cls, v):
        if v not in VALID_KLINES:
            raise ValueError(f"Invalid kline: {v}")
        return v

    @field_validator("target")
    @classmethod
    def validate_target(cls, v):
        if len(v) == 0:
            raise ValueError("At least one target ticker is required")
        return v

    @field_validator("label")
    @classmethod
    def validate_label(cls, v):
        if not v.strip():
            raise ValueError("Label expression cannot be empty")
        return v

    @field_validator("training_pool")
    @classmethod
    def validate_training_pool(cls, v):
        if v not in VALID_TRAINING_POOLS:
            raise ValueError(f"Invalid training pool: {v}")
        return v


# ===== Response Models =====

class TradeRecord(BaseModel):
    ticker: str = ""
    buy_date: str
    buy_price: Optional[float] = None
    sell_date: str
    sell_price: Optional[float] = None
    days: int
    return_pct: Optional[float] = None
    is_open: bool = False


class SegmentRecord(BaseModel):
    ticker: str = ""
    type: str  # "hold" or "empty"
    start: str
    end: str
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    days: int
    change_pct: Optional[float] = None


class SummaryStats(BaseModel):
    model_return: float
    bh_return: float
    win_rate: float
    total_trades: int
    test_start: str
    test_end: str
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    held_days: int = 0
    total_days: int = 0


class HoldingStats(BaseModel):
    count: int
    total_days: int
    profit_segs: int
    loss_segs: int
    avg_return: float


class EmptyStats(BaseModel):
    count: int
    total_days: int
    missed_up: float
    avoided_down: float


class TimelineSegment(BaseModel):
    type: str  # "hold" or "empty"
    pct: float


class BacktestResult(BaseModel):
    summary: SummaryStats
    trades: List[TradeRecord]
    segments: List[SegmentRecord]
    holding_stats: Optional[HoldingStats] = None
    empty_stats: Optional[EmptyStats] = None
    timeline: List[TimelineSegment] = []


# ===== Scan Models =====

class ScanRequest(BaseModel):
    ticker: str
    data_source: str = "db"
    trading_style: str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        if not v.strip():
            raise ValueError("Ticker cannot be empty")
        return v.upper().strip()

    @field_validator("data_source")
    @classmethod
    def validate_data_source(cls, v):
        if v not in VALID_DATA_SOURCES:
            raise ValueError(f"Invalid data source: {v}")
        return v

    @field_validator("trading_style")
    @classmethod
    def validate_trading_style(cls, v):
        if v not in ("ultra_short", "swing"):
            raise ValueError("Must be 'ultra_short' or 'swing'")
        return v


class PredictionSignal(BaseModel):
    date: str
    score: float
    signal: str  # "buy" | "sell" | "hold"


class PredictionResult(BaseModel):
    ticker: str
    model_class: str
    prediction_range: Tuple[str, str]
    signals: List[PredictionSignal]
    latest_signal: str
    latest_score: float
    threshold: float
    generated_at: str

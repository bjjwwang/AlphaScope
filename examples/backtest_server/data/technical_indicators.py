"""
技术指标计算模块

包含以下指标:
- 移动平均线 (MA)
- MACD (Moving Average Convergence Divergence)
- 布林带 (Bollinger Bands)
- Hurst 带 (近似实现)
- 成交量比率 (Volume Ratio)
- 换手率 (Turnover Rate)
- 斜率计算 (Slope)
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def calculate_ma(close: pd.Series, period: int) -> pd.Series:
    """计算移动平均线"""
    return close.rolling(window=period).mean()


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """计算指数移动平均线"""
    return close.ewm(span=period, adjust=False).mean()


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    计算 MACD

    Returns:
        dict with 'macd', 'signal', 'histogram'
    """
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    计算布林带

    Returns:
        dict with 'upper', 'middle', 'lower'
    """
    middle = calculate_ma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def calculate_hurst_bands(close: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
    """
    计算 Hurst 带 (近似实现)

    Hurst 带基于价格的波动性和趋势方向
    这里使用简化版本: 基于 ATR 和趋势的动态通道

    Returns:
        dict with 'outer_upper', 'inner_upper', 'middle', 'inner_lower', 'outer_lower'
    """
    # 计算中轨 (EMA)
    middle = calculate_ema(close, period)

    # 计算波动性 (标准差)
    std = close.rolling(window=period).std()

    # 内轨 (1倍标准差)
    inner_upper = middle + std
    inner_lower = middle - std

    # 外轨 (2倍标准差)
    outer_upper = middle + (std * 2)
    outer_lower = middle - (std * 2)

    return {
        'outer_upper': outer_upper,
        'inner_upper': inner_upper,
        'middle': middle,
        'inner_lower': inner_lower,
        'outer_lower': outer_lower
    }


def calculate_volume_ratio(volume: pd.Series, period: int = 5) -> pd.Series:
    """
    计算量比

    量比 = 当前成交量 / 过去 N 日平均成交量
    """
    avg_volume = volume.rolling(window=period).mean().shift(1)
    return volume / avg_volume.replace(0, float("nan"))


def calculate_slope(series: pd.Series, period: int = 5) -> pd.Series:
    """
    计算斜率 (角度)

    使用线性回归计算斜率，并转换为角度
    """
    def slope_degrees(x):
        if len(x) < 2 or x.isna().any():
            return np.nan
        y = np.arange(len(x))
        try:
            slope = np.polyfit(y, x.values, 1)[0]
            # 归一化斜率 (相对于价格的百分比变化)
            if x.iloc[0] == 0:
                return np.nan
            normalized_slope = slope / x.iloc[0] * 100
            # 转换为角度
            angle = np.degrees(np.arctan(normalized_slope))
            return angle
        except:
            return np.nan

    return series.rolling(window=period).apply(slope_degrees, raw=False)


def detect_macd_divergence(close: pd.Series, macd: pd.Series, lookback: int = 20) -> pd.Series:
    """
    检测 MACD 背离

    返回:
        1: 底背离 (价格新低但 MACD 未新低)
        -1: 顶背离 (价格新高但 MACD 未新高)
        0: 无背离
    """
    result = pd.Series(0, index=close.index)

    for i in range(lookback, len(close)):
        window_close = close.iloc[i-lookback:i+1]
        window_macd = macd.iloc[i-lookback:i+1]

        # 检测底背离: 价格创新低，但 MACD 没有创新低
        if close.iloc[i] == window_close.min():
            prev_low_idx = window_close.idxmin()
            if prev_low_idx != close.index[i]:
                prev_macd = macd.loc[prev_low_idx]
                if macd.iloc[i] > prev_macd:
                    result.iloc[i] = 1  # 底背离

        # 检测顶背离: 价格创新高，但 MACD 没有创新高
        if close.iloc[i] == window_close.max():
            prev_high_idx = window_close.idxmax()
            if prev_high_idx != close.index[i]:
                prev_macd = macd.loc[prev_high_idx]
                if macd.iloc[i] < prev_macd:
                    result.iloc[i] = -1  # 顶背离

    return result


def detect_golden_cross(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    """
    检测金叉

    Returns:
        True when fast_ma crosses above slow_ma
    """
    prev_fast = fast_ma.shift(1)
    prev_slow = slow_ma.shift(1)

    # 金叉: 之前 fast < slow, 现在 fast >= slow
    return (prev_fast < prev_slow) & (fast_ma >= slow_ma)


def detect_death_cross(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    """
    检测死叉

    Returns:
        True when fast_ma crosses below slow_ma
    """
    prev_fast = fast_ma.shift(1)
    prev_slow = slow_ma.shift(1)

    # 死叉: 之前 fast > slow, 现在 fast <= slow
    return (prev_fast > prev_slow) & (fast_ma <= slow_ma)


def is_ma_bullish_alignment(ma5: float, ma10: float, ma20: float) -> bool:
    """检测均线多头排列: MA5 > MA10 > MA20"""
    return ma5 > ma10 > ma20


def is_price_above_bollinger_mid(close: float, bollinger_middle: float) -> bool:
    """价格在布林带中轨之上"""
    return close > bollinger_middle


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标

    Input DataFrame 需要包含: open, high, low, close, volume

    Returns:
        DataFrame with all indicators added
    """
    result = df.copy()

    # 移动平均线
    result['ma5'] = calculate_ma(df['close'], 5)
    result['ma10'] = calculate_ma(df['close'], 10)
    result['ma20'] = calculate_ma(df['close'], 20)
    result['ma60'] = calculate_ma(df['close'], 60)

    # MACD
    macd = calculate_macd(df['close'])
    result['macd'] = macd['macd']
    result['macd_signal'] = macd['signal']
    result['macd_histogram'] = macd['histogram']

    # 布林带
    bollinger = calculate_bollinger_bands(df['close'])
    result['bb_upper'] = bollinger['upper']
    result['bb_middle'] = bollinger['middle']
    result['bb_lower'] = bollinger['lower']

    # Hurst 带
    hurst = calculate_hurst_bands(df['close'])
    result['hurst_outer_upper'] = hurst['outer_upper']
    result['hurst_inner_upper'] = hurst['inner_upper']
    result['hurst_middle'] = hurst['middle']
    result['hurst_inner_lower'] = hurst['inner_lower']
    result['hurst_outer_lower'] = hurst['outer_lower']

    # 量比
    result['volume_ratio'] = calculate_volume_ratio(df['volume'])

    # 斜率
    result['hurst_outer_slope'] = calculate_slope(hurst['outer_upper'], 5)
    result['hurst_inner_slope'] = calculate_slope(hurst['inner_upper'], 5)

    # MACD 背离
    result['macd_divergence'] = detect_macd_divergence(df['close'], macd['macd'])

    # 金叉/死叉
    result['ma5_10_golden'] = detect_golden_cross(result['ma5'], result['ma10'])
    result['macd_golden'] = detect_golden_cross(macd['macd'], macd['signal'])

    return result

"""
5-Phase Stock Filter (Mutually Exclusive)

Based on quantflow-ai project defined 5 market phases, mutually exclusive:
1. bottom: Price near Hurst lower band, MACD below zero, volume shrinking
2. inflection: Just rising from bottom, MACD below zero but improving, price just above middle band
3. moderate: Steady uptrend, MACD above zero, slope 20°-45°
4. strong: Accelerating uptrend, steep slope (>45°), volume expanding
5. weekly_strong_daily_flat: Weekly strong but daily consolidating

Priority order: bottom > inflection > weekly_strong_daily_flat > moderate > strong
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from technical_indicators import (
    calculate_all_indicators,
    calculate_hurst_bands,
    calculate_slope,
    is_ma_bullish_alignment,
    is_price_above_bollinger_mid
)


class PhaseFilter:
    """Base class for phase filters"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def check(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> Tuple[bool, str]:
        raise NotImplementedError


class BottomPhaseFilter(PhaseFilter):
    """
    Bottom Phase Filter (Most depressed stage)

    Core characteristics:
    - Price near or below Hurst outer lower band
    - MACD below zero
    - Volume shrinking or bullish divergence signal
    """

    def __init__(self):
        super().__init__(
            'bottom',
            'Bottom: Price at lower band, MACD below zero, low volume or divergence'
        )

    def check(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> Tuple[bool, str]:
        if len(daily_df) < 30:
            return False, "Insufficient data"

        latest = daily_df.iloc[-1]
        reasons = []

        # Core condition 1: Price near Hurst outer lower band
        price_near_lower = latest['close'] <= latest['hurst_outer_lower'] * 1.03
        if not price_near_lower:
            return False, "Price not at lower band"
        reasons.append("Price at Hurst outer lower band")

        # Core condition 2: MACD below zero (confirming weakness)
        macd_below_zero = latest['macd'] < 0
        if not macd_below_zero:
            return False, "MACD not below zero"
        reasons.append("MACD below zero")

        # Supporting conditions (at least one): Low volume or MACD divergence or double golden cross below zero
        low_volume = latest['volume_ratio'] < 0.8 if pd.notna(latest['volume_ratio']) else False
        has_divergence = latest['macd_divergence'] == 1

        recent = daily_df.tail(20)
        golden_crosses = recent[recent['macd_golden'] == True]
        below_zero_crosses = golden_crosses[golden_crosses['macd'] < 0]
        has_double_golden = len(below_zero_crosses) >= 2

        if low_volume:
            reasons.append(f"Low volume ratio ({latest['volume_ratio']:.2f})")
        if has_divergence:
            reasons.append("MACD bullish divergence")
        if has_double_golden:
            reasons.append("MACD double golden cross below zero")

        has_signal = low_volume or has_divergence or has_double_golden
        if not has_signal:
            return False, "No bottom signal"

        return True, "; ".join(reasons)


class InflectionPhaseFilter(PhaseFilter):
    """
    Inflection Phase Filter (Just rising from bottom)

    Core characteristics:
    - MACD still below zero but improving
    - Price just crossed or above Hurst middle band (within last 3 days)
    - MA5 crossing above MA10 (short-term trend improving)

    Exclusion conditions:
    - Price already far from middle band (>5% above)
    - MACD already above zero (should be moderate/strong)
    """

    def __init__(self):
        super().__init__(
            'inflection',
            'Inflection: MACD below zero improving, price just above middle band, MA5 cross MA10'
        )

    def check(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> Tuple[bool, str]:
        if len(daily_df) < 30:
            return False, "Insufficient data"

        latest = daily_df.iloc[-1]
        prev = daily_df.iloc[-2]
        reasons = []

        # Exclusion: MACD already above zero (should be moderate or strong)
        if latest['macd'] >= 0:
            return False, "MACD above zero (should be moderate/strong)"

        # Core condition 1: Price near Hurst middle band (just crossed, not too far)
        above_middle = latest['close'] >= latest['hurst_middle']
        not_too_far = latest['close'] <= latest['hurst_middle'] * 1.05
        cross_middle = (prev['close'] < prev['hurst_middle']) and (latest['close'] >= latest['hurst_middle'])

        if cross_middle:
            reasons.append("Price crossed Hurst middle band")
        elif above_middle and not_too_far:
            reasons.append("Price just above Hurst middle band")
        else:
            return False, "Price position not inflection pattern"

        # Core condition 2: MACD improving (histogram increasing)
        macd_improving = latest['macd_histogram'] > prev['macd_histogram']
        if macd_improving:
            reasons.append("MACD improving from low")

        # Core condition 3: MA5 crossing MA10 (within last 3 days)
        recent_cross = daily_df.tail(3)['ma5_10_golden'].any()
        if recent_cross:
            reasons.append("MA5 crossed above MA10")

        # Need: position condition + (MACD improving or MA cross)
        momentum_condition = macd_improving or recent_cross
        if not momentum_condition:
            return False, "No inflection momentum signal"

        return True, "; ".join(reasons)


class ModeratePhaseFilter(PhaseFilter):
    """
    Moderate Ascent Phase Filter

    Core characteristics:
    - MACD above zero
    - MA20 providing support
    - Hurst outer band slope 20°-45° (steady rise, not steep)

    Exclusion conditions:
    - Slope > 45° (should be strong)
    - Slope < 20° (trend not clear)
    """

    def __init__(self):
        super().__init__(
            'moderate',
            'Moderate Ascent: MACD above zero, MA20 support, slope 20°-45°'
        )

    def check(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> Tuple[bool, str]:
        if len(daily_df) < 30:
            return False, "Insufficient data"

        latest = daily_df.iloc[-1]
        reasons = []

        # Core condition 1: MACD above zero
        macd_above_zero = latest['macd'] > 0
        if not macd_above_zero:
            return False, "MACD not above zero"
        reasons.append("MACD above zero")

        # Core condition 2: MA20 effective support (price above MA20 for last 5 days)
        above_ma20 = latest['close'] > latest['ma20']
        ma20_support = (daily_df.tail(5)['close'] > daily_df.tail(5)['ma20']).all()
        if not (above_ma20 and ma20_support):
            return False, "MA20 support unstable"
        reasons.append("MA20 effective support")

        # Core condition 3: Slope check (distinguish moderate vs strong)
        slope = latest['hurst_outer_slope']
        if pd.isna(slope):
            return False, "Insufficient slope data"

        # Moderate ascent: slope 20°-45°
        if slope > 45:
            return False, f"Slope too steep ({slope:.1f}°), should be strong"
        if slope < 20:
            return False, f"Slope too flat ({slope:.1f}°), trend unclear"

        reasons.append(f"Moderate slope ({slope:.1f}°)")

        return True, "; ".join(reasons)


class StrongPhaseFilter(PhaseFilter):
    """
    Strong Ascent Phase Filter

    Core characteristics:
    - MA 5>10>20 bullish alignment
    - MACD above zero and strong (histogram > 0)
    - Hurst outer band slope > 45° (steep rise)
    - Volume ratio expanding (> 1.2) or breaking previous high
    """

    def __init__(self):
        super().__init__(
            'strong',
            'Strong Ascent: MA bullish alignment, slope >45°, volume expanding'
        )

    def check(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> Tuple[bool, str]:
        if len(daily_df) < 30:
            return False, "Insufficient data"

        latest = daily_df.iloc[-1]
        reasons = []

        # Core condition 1: MA bullish alignment (MA5 > MA10 > MA20)
        ma_bullish = is_ma_bullish_alignment(latest['ma5'], latest['ma10'], latest['ma20'])
        if not ma_bullish:
            return False, "MA not bullish aligned"
        reasons.append("MA bullish alignment (5>10>20)")

        # Core condition 2: MACD strong (above zero and histogram > 0)
        macd_strong = latest['macd'] > 0 and latest['macd_histogram'] > 0
        if not macd_strong:
            return False, "MACD not strong enough"
        reasons.append("MACD strong")

        # Core condition 3: Steep slope (> 45°) indicates acceleration
        slope = latest['hurst_outer_slope']
        if pd.isna(slope) or slope <= 45:
            return False, f"Slope not steep enough ({slope:.1f}° <= 45°)"
        reasons.append(f"Steep slope ({slope:.1f}°)")

        # Supporting conditions: Volume expanding or breaking previous high
        high_volume = latest['volume_ratio'] > 1.2 if pd.notna(latest['volume_ratio']) else False
        recent_20 = daily_df.tail(20)
        is_new_high = latest['close'] >= recent_20['close'].max() * 0.98

        if high_volume:
            reasons.append(f"Volume expanding ({latest['volume_ratio']:.2f})")
        if is_new_high:
            reasons.append("Breaking 20-day high")

        return True, "; ".join(reasons)


class WeeklyStrongDailyFlatPhaseFilter(PhaseFilter):
    """
    Weekly Strong Daily Flat Phase Filter

    Core characteristics:
    - Weekly Hurst inner band slope > 45° (weekly strong uptrend)
    - Daily Hurst outer band flat (-15° to 15°) or daily range narrowing (< 8%)
    - Daily MACD above zero but not accelerating (waiting for breakout)

    This is a special consolidation phase, building energy for next wave up
    """

    def __init__(self):
        super().__init__(
            'weekly_strong_daily_flat',
            'Weekly Strong Daily Flat: Weekly slope >45°, daily consolidating'
        )

    def check(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> Tuple[bool, str]:
        if len(daily_df) < 30:
            return False, "Insufficient daily data"

        if weekly_df is None or len(weekly_df) < 10:
            return False, "Insufficient weekly data"

        daily_latest = daily_df.iloc[-1]
        weekly_latest = weekly_df.iloc[-1]
        reasons = []

        # Core condition 1: Weekly Hurst inner band slope > 45° (weekly strong)
        weekly_slope = weekly_latest.get('hurst_inner_slope', None)
        if weekly_slope is None:
            weekly_hurst = calculate_hurst_bands(weekly_df['close'])
            weekly_slope_series = calculate_slope(weekly_hurst['inner_upper'], 5)
            weekly_slope = weekly_slope_series.iloc[-1] if len(weekly_slope_series) > 0 else None

        if pd.isna(weekly_slope) or weekly_slope <= 45:
            return False, f"Weekly slope not strong enough ({weekly_slope:.1f}° <= 45°)"
        reasons.append(f"Weekly slope strong ({weekly_slope:.1f}°)")

        # Core condition 2: Daily flat or consolidating
        daily_slope = daily_latest['hurst_outer_slope']
        daily_flat = pd.notna(daily_slope) and -15 <= daily_slope <= 15

        recent_10 = daily_df.tail(10)
        price_range = (recent_10['high'].max() - recent_10['low'].min()) / recent_10['close'].mean()
        consolidating = price_range < 0.08  # Range < 8%

        if not (daily_flat or consolidating):
            return False, "Daily not flat or consolidating"

        if daily_flat:
            reasons.append(f"Daily flat ({daily_slope:.1f}°)")
        if consolidating:
            reasons.append(f"Daily consolidating (range {price_range*100:.1f}%)")

        # Exclusion: Daily slope > 30° means daily also accelerating, not consolidating
        if pd.notna(daily_slope) and daily_slope > 30:
            return False, "Daily also accelerating, not consolidating"

        return True, "; ".join(reasons)


def get_all_filters() -> dict:
    """Get all phase filters"""
    return {
        'bottom': BottomPhaseFilter(),
        'inflection': InflectionPhaseFilter(),
        'moderate': ModeratePhaseFilter(),
        'strong': StrongPhaseFilter(),
        'weekly_strong_daily_flat': WeeklyStrongDailyFlatPhaseFilter()
    }


def get_filter(phase: str) -> PhaseFilter:
    """Get filter for specified phase"""
    filters = get_all_filters()
    if phase not in filters:
        raise ValueError(f"Unknown phase: {phase}. Available: {list(filters.keys())}")
    return filters[phase]


def classify_stock(daily_df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> Tuple[Optional[str], str]:
    """
    Classify a stock into a phase (mutually exclusive)

    Priority order: bottom > inflection > weekly_strong_daily_flat > moderate > strong

    Returns:
        (phase, reason): Phase and reason, or (None, "...") if no phase matches
    """
    filters = get_all_filters()

    # Check in priority order
    priority_order = ['bottom', 'inflection', 'weekly_strong_daily_flat', 'moderate', 'strong']

    for phase in priority_order:
        filter_obj = filters[phase]
        is_match, reason = filter_obj.check(daily_df, weekly_df)
        if is_match:
            return phase, reason

    return None, "Does not match any phase criteria"

from src.strategy_base import BaseStrategy
from src.strategy_registry import StrategyRegistry
import pandas as pd
from typing import Dict, Any

@StrategyRegistry.register
class MACrossStrategy(BaseStrategy):
    @property
    def strategy_info(self):
        return {
            'name': 'MA Cross Strategy',
            'description': 'Moving Average Crossover Strategy',
            'timeframes': ['1h', '4h', '1d'],
            'parameters': {
                'fast_ma': 'Fast Moving Average period',
                'slow_ma': 'Slow Moving Average period'
            }
        }
    
    @property
    def parameter_space(self):
        return {
            'fast_ma': {
                'min': 3,
                'max': 50,
                'step': 1,
                'type': 'int'
            },
            'slow_ma': {
                'min': 10,
                'max': 100,
                'step': 1,
                'type': 'int'
            }
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        fast_ma = self.parameters.get('fast_ma', 10)
        slow_ma = self.parameters.get('slow_ma', 20)
        
        # Basit hareketli ortalama (SMA) hesaplama
        df['fast_ma'] = df['close'].rolling(window=fast_ma).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_ma).mean()
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['signal'] = 0
        
        # Trend yönünü belirle
        df['trend'] = df['fast_ma'] - df['slow_ma']
        df['prev_trend'] = df['trend'].shift(1)
        
        # Long sinyalleri
        df.loc[(df['prev_trend'] < 0) & (df['trend'] > 0), 'signal'] = 1
        
        # Short sinyalleri
        df.loc[(df['prev_trend'] > 0) & (df['trend'] < 0), 'signal'] = -1
        
        return df

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """MA Cross için parametre validasyonu"""
        fast_ma = params.get('fast_ma', 0)
        slow_ma = params.get('slow_ma', 0)
        return fast_ma < slow_ma  # fast_ma her zaman slow_ma'dan küçük olmalı
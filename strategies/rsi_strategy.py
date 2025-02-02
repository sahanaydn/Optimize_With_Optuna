from src.strategy_base import BaseStrategy
from src.strategy_registry import StrategyRegistry
import pandas as pd
from typing import Dict, Any

@StrategyRegistry.register
class RSIStrategy(BaseStrategy):
    @property
    def strategy_info(self):
        return {
            'name': 'RSI Strategy',
            'description': 'RSI Oversold/Overbought Strategy',
            'timeframes': ['1h', '4h', '1d'],
            'parameters': {
                'rsi_period': 'RSI hesaplama periyodu',
                'oversold': 'Aşırı satım seviyesi',
                'overbought': 'Aşırı alım seviyesi'
            }
        }
    
    @property
    def parameter_space(self):
        return {
            'rsi_period': {
                'min': 2,
                'max': 50,
                'step': 1,
                'type': 'int'
            },
            'oversold': {
                'min': 10,
                'max': 45,
                'step': 1,
                'type': 'int'
            },
            'overbought': {
                'min': 55,
                'max': 90,
                'step': 1,
                'type': 'int'
            }
        }
    
    def calculate_rsi(self, prices, period=14):
        deltas = (prices - prices.shift(1)).dropna()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rsi_period = self.parameters.get('rsi_period', 14)
        
        # RSI hesapla (talib yerine kendi fonksiyonumuzu kullan)
        df['rsi'] = self.calculate_rsi(df['close'], period=rsi_period)
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['signal'] = 0
        
        oversold = self.parameters.get('oversold', 30)
        overbought = self.parameters.get('overbought', 70)
        
        # Long sinyalleri (RSI aşırı satım seviyesinden yukarı çıkınca)
        df.loc[(df['rsi'] > oversold) & (df['rsi'].shift(1) <= oversold), 'signal'] = 1
        
        # Short sinyalleri (RSI aşırı alım seviyesinden aşağı inince)
        df.loc[(df['rsi'] < overbought) & (df['rsi'].shift(1) >= overbought), 'signal'] = -1
        
        return df

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """RSI için parametre validasyonu"""
        oversold = params.get('oversold', 0)
        overbought = params.get('overbought', 100)
        return oversold < overbought  # oversold her zaman overbought'tan küçük olmalı 
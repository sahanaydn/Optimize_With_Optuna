from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
import pandas as pd
import logging
import numpy as np

@dataclass
class StrategyParameters:
    """Strateji parametrelerini tutan veri sınıfı"""
    name: str
    timeframe: str
    parameters: Dict[str, Any]

class BaseStrategy(ABC):
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.position = 0
        self.trades = []
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """İndikatörleri hesapla"""
        pass
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alım-satım sinyallerini üret"""
        pass
        
    @property
    @abstractmethod
    def strategy_info(self) -> Dict[str, Any]:
        """
        Strateji meta bilgileri
        Returns:
            Dict: {
                'name': str - Strateji adı
                'description': str - Strateji açıklaması
                'timeframes': List[str] - Desteklenen zaman dilimleri
                'parameters': Dict[str, str] - Parametre açıklamaları
            }
        """
        pass
        
    @property
    @abstractmethod
    def parameter_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Optimizasyon için parametre uzayı
        Returns:
            Dict: {
                'param_name': {
                    'min': number,
                    'max': number,
                    'step': number,
                    'type': 'int' | 'float'
                }
            }
        """
        pass

    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Parametre kombinasyonunun geçerli olup olmadığını kontrol et
        Returns:
            bool: Parametreler geçerliyse True, değilse False
        """
        pass

    def execute_trades(self, df: pd.DataFrame) -> List[Dict]:
        """
        Sinyallere göre işlemleri gerçekleştir
        Bu metod tüm stratejiler için ortak, override edilmemeli
        """
        try:
            signals = self.generate_signals(df)
            trades = []
            position = None
            
            for i in range(1, len(signals)):
                current_time = signals.index[i]
                current_price = signals['close'].iloc[i]
                signal = signals['signal'].iloc[i]
                
                # Pozisyon kapatma
                if position and (
                    (position['type'] == 'long' and signal == -1) or
                    (position['type'] == 'short' and signal == 1)
                ):
                    pnl = self._calculate_pnl(position, current_price)
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'type': position['type'],
                        'pnl': pnl
                    })
                    position = None
                
                # Yeni pozisyon açma
                if not position and signal != 0:
                    position = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'type': 'long' if signal == 1 else 'short'
                    }
            
            # Son pozisyonu kapat
            if position:
                last_price = signals['close'].iloc[-1]
                pnl = self._calculate_pnl(position, last_price)
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': signals.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'type': position['type'],
                    'pnl': pnl
                })
            
            self.trades = trades
            return trades
            
        except Exception as e:
            self.logger.error(f"Error executing trades: {str(e)}")
            raise

    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """PnL hesapla"""
        if position['type'] == 'long':
            return (exit_price - position['entry_price']) / position['entry_price']
        return (position['entry_price'] - exit_price) / position['entry_price']

    def calculate_performance(self, initial_capital: float = 1000) -> Dict[str, float]:
        """Strateji performans metriklerini hesapla"""
        trades = self.trades
        
        if not trades:
            return {
                'total_return': 0,
                'compound_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
        
        # Sabit sermaye ile return (1000$)
        total_pnl = sum(trade['pnl'] for trade in trades)
        total_return = total_pnl  # Yüzde olarak zaten
        
        # Bileşik return (tüm sermaye ile)
        compound_return = -1
        if trades:
            compound_return = (1 + pd.Series([t['pnl'] for t in trades])).prod() - 1
        
        # Diğer metrikler aynı kalır...
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit Factor hesapla
        profits = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
        profit_factor = sum(profits) / sum(losses) if losses else 0
        
        # Drawdown ve Sharpe hesapla...
        returns = pd.Series([t['pnl'] for t in trades])
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0
        
        sharpe_ratio = 0
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
        
        return {
            'total_return': total_return,  # Sabit sermaye ile
            'compound_return': compound_return,  # Bileşik return
            'total_trades': total_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor
        } 
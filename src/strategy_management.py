from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

@dataclass
class StrategyParameters:
    """Strateji parametrelerini tutan veri sınıfı"""
    name: str
    timeframe: str
    parameters: Dict[str, Any]

class BaseStrategy:
    """Temel strateji sınıfı - tüm stratejiler bu sınıftan türetilecek"""
    
    def __init__(self, parameters: StrategyParameters):
        self.name = parameters.name
        self.timeframe = parameters.timeframe
        self.parameters = parameters.parameters
        self.logger = logging.getLogger(__name__)
        
        # Pozisyon ve işlem takibi için değişkenler
        self.position = 0  # 1: Long, -1: Short, 0: Kapalı
        self.trades: List[Dict] = []
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Teknik indikatörleri hesapla
        Her strateji kendi indikatörlerini implement edecek
        """
        raise NotImplementedError("Her strateji kendi indikatörlerini tanımlamalı")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alım-satım sinyallerini üret
        Her strateji kendi sinyal mantığını implement edecek
        """
        raise NotImplementedError("Her strateji kendi sinyal mantığını tanımlamalı")
    
    def execute_trades(self, df: pd.DataFrame) -> List[Dict]:
        """Sinyallere göre işlemleri gerçekleştir"""
        try:
            signals = self.generate_signals(df)
            if signals.empty:
                raise ValueError("No signals generated")
            
            trades = []
            position = None
            
            print("\nProcessing signals...")
            
            for i in range(1, len(signals)):
                current_time = signals.index[i]
                current_price = signals['close'].iloc[i]
                signal = signals['signal'].iloc[i]
                
                # Debug için her adımı yazdır
                print(f"\nStep {i}:")
                print(f"Time: {current_time}")
                print(f"Signal: {signal}")
                print(f"Current Position: {position['type'] if position else 'None'}")
                
                # Mevcut pozisyonu kapat (eğer ters sinyal varsa)
                if position is not None:
                    should_close = False
                    
                    if position['type'] == 'long' and signal == -1:
                        should_close = True
                        pnl = (current_price - position['entry_price']) / position['entry_price']
                    elif position['type'] == 'short' and signal == 1:
                        should_close = True
                        pnl = (position['entry_price'] - current_price) / position['entry_price']
                    
                    if should_close:
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'type': position['type'],
                            'pnl': pnl
                        })
                        print(f"Closing {position['type'].upper()} position at {current_time}, PnL: {pnl:.2%}")
                        position = None
                
                # Yeni pozisyon aç (eğer sinyal varsa ve pozisyon yoksa)
                if position is None and signal != 0:
                    position_type = 'long' if signal == 1 else 'short'
                    position = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'type': position_type
                    }
                    print(f"Opening {position_type.upper()} position at {current_time}, price: {current_price}")
            
            # Son pozisyonu kapat
            if position is not None:
                last_time = signals.index[-1]
                last_price = signals['close'].iloc[-1]
                
                if position['type'] == 'long':
                    pnl = (last_price - position['entry_price']) / position['entry_price']
                else:  # short
                    pnl = (position['entry_price'] - last_price) / position['entry_price']
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': last_time,
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'type': position['type'],
                    'pnl': pnl
                })
                print(f"Closing final {position['type'].upper()} position at {last_time}, PnL: {pnl:.2%}")
            
            # Trade istatistiklerini yazdır
            print("\nTrade Summary:")
            long_trades = len([t for t in trades if t['type'] == 'long'])
            short_trades = len([t for t in trades if t['type'] == 'short'])
            print(f"Total trades: {len(trades)}")
            print(f"Long trades: {long_trades}")
            print(f"Short trades: {short_trades}")
            
            self.trades = trades
            return trades
        
        except Exception as e:
            self.logger.error(f"Error executing trades: {str(e)}")
            raise
    
    def calculate_performance(self) -> Dict[str, float]:
        """Strateji performans metriklerini hesapla"""
        trades = self.trades  # trades listesini saklıyoruz
        
        if not trades:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'risk_reward_ratio': 0
            }
            
        profits = [trade['pnl'] for trade in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        total_return = np.prod([1 + p for p in profits]) - 1
        win_rate = len(winning_trades) / len(profits) if profits else 0
        
        profit_factor = (
            abs(sum(winning_trades) / sum(losing_trades))
            if losing_trades and winning_trades else 0
        )
        
        return {
            'total_return': total_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': np.mean(profits),
            'max_drawdown': self._calculate_max_drawdown(profits),
            'sharpe_ratio': self._calculate_sharpe_ratio(profits),
            'sortino_ratio': self._calculate_sortino_ratio(profits),
            'calmar_ratio': self._calculate_calmar_ratio(total_return, profits),
            'avg_win': np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if losing_trades else 0,
            'max_consecutive_wins': self._calculate_max_consecutive(profits, True),
            'max_consecutive_losses': self._calculate_max_consecutive(profits, False),
            'risk_reward_ratio': abs(np.mean(winning_trades) / np.mean(losing_trades)) if losing_trades else 0
        }
    
    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """Maksimum drawdown hesapla"""
        cumulative = np.cumsum(profits)
        max_dd = 0
        peak = cumulative[0]
        
        for value in cumulative:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, profits: List[float], risk_free_rate: float = 0.02) -> float:
        """Sharpe oranı hesapla"""
        if not profits:
            return 0
        
        returns = np.array(profits)
        excess_returns = returns - risk_free_rate/252  # Günlük risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0
            
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self, profits: List[float], risk_free_rate: float = 0.02) -> float:
        """Sortino oranını hesapla"""
        if not profits:
            return 0
            
        returns = np.array(profits)
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0
            
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        
    def _calculate_calmar_ratio(self, total_return: float, profits: List[float]) -> float:
        """Calmar oranını hesapla"""
        max_dd = self._calculate_max_drawdown(profits)
        if max_dd == 0:
            return 0
            
        return total_return / max_dd

    def _calculate_max_consecutive(self, profits: List[float], is_win: bool) -> int:
        """Maksimum ardışık kazanç/kaybetme sayısını hesapla"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in profits:
            if (pnl > 0 and is_win) or (pnl <= 0 and not is_win):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive

class MovingAverageCrossStrategy(BaseStrategy):
    """Örnek bir strateji: Hareketli Ortalama Kesişimi"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """İki farklı periyotlu hareketli ortalama hesapla"""
        df = df.copy()
        
        fast_ma = self.parameters.get('fast_ma', 10)
        slow_ma = self.parameters.get('slow_ma', 20)
        
        df['fast_ma'] = df['close'].rolling(window=fast_ma).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_ma).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hareketli ortalama kesişimlerine göre sinyaller üret"""
        df = self.calculate_indicators(df)
        
        # Başlangıçta sinyali 0 olarak ayarla
        df['signal'] = 0
        
        # Trend yönünü belirle
        df['trend'] = df['fast_ma'] - df['slow_ma']
        df['prev_trend'] = df['trend'].shift(1)
        
        # Long sinyalleri: Trend pozitife dönüyor
        df.loc[(df['prev_trend'] < 0) & (df['trend'] > 0), 'signal'] = 1
        
        # Short sinyalleri: Trend negatife dönüyor
        df.loc[(df['prev_trend'] > 0) & (df['trend'] < 0), 'signal'] = -1
        
        # Temizlik
        df = df.drop(['trend', 'prev_trend'], axis=1)
        
        return df 
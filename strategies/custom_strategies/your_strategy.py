from src.strategy_management import BaseStrategy
import pandas as pd

class YourStrategy(BaseStrategy):
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
        
        # Debug için trend değişimlerini yazdır
        trend_changes = df[df['signal'] != 0]
        print("\nTrend Changes:")
        for idx, row in trend_changes.iterrows():
            signal_type = "LONG" if row['signal'] == 1 else "SHORT"
            print(f"Time: {idx}, Signal: {signal_type}")
            print(f"Fast MA: {row['fast_ma']:.2f}, Slow MA: {row['slow_ma']:.2f}")
            print(f"Trend: {row['trend']:.2f}, Prev Trend: {row['prev_trend']:.2f}\n")
        
        # Temizlik
        df = df.drop(['trend', 'prev_trend'], axis=1)
        
        return df 
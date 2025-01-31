import ccxt
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional

class DataManager:
    def __init__(self, exchange: str = 'binance'):
        """
        Veri yönetimi için ana sınıf
        
        Args:
            exchange: Borsa adı (default: 'binance')
        """
        self.exchange = getattr(ccxt, exchange)()
        self.logger = logging.getLogger(__name__)
        
    def fetch_ohlcv(self,
                   symbol: str,
                   timeframe: str = '1h',
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        OHLCV verilerini çek
        
        Args:
            symbol: Trading pair (örn: 'BTC/USDT')
            timeframe: Zaman dilimi (örn: '1h', '4h', '1d')
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            
        Returns:
            pd.DataFrame: OHLCV verileri
        """
        try:
            # Tarihleri milisaniyeye çevir
            since = int(start_date.timestamp() * 1000) if start_date else None
            until = int(end_date.timestamp() * 1000) if end_date else None
            
            all_ohlcv = []
            current_since = since
            
            while True:
                # Verileri çek
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Son çekilen verinin timestamp'i
                last_timestamp = ohlcv[-1][0]
                
                # Eğer bitiş tarihine ulaştıysak veya yeterli veri çektiyse dur
                if until and last_timestamp >= until:
                    break
                    
                # Bir sonraki sorgu için timestamp'i güncelle
                current_since = last_timestamp + 1
                
                # Rate limit'e takılmamak için bekle
                self.exchange.sleep(self.exchange.rateLimit / 1000)
            
            # DataFrame'e çevir
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Timestamp'i datetime'a çevir
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Tarih filtreleme
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            self.logger.error(f"Data fetch error: {str(e)}")
            raise 
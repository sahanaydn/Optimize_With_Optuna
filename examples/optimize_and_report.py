import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.optimization import StrategyOptimizer
from src.reporting import ReportGenerator
from strategies.custom_strategies.your_strategy import YourStrategy
import ccxt
import pandas as pd
from datetime import datetime, timedelta

# Binance'den veri çek
exchange = ccxt.binance({
    'timeout': 30000,  # 30 saniye
    'enableRateLimit': True,  # Rate limiting aktif
    'rateLimit': 1000  # Her istek arası 1 saniye bekle
})

# Hata yönetimi ekleyelim
max_retries = 3
retry_count = 0

while retry_count < max_retries:
    try:
        end_time = exchange.milliseconds()
        start_time = end_time - (90 * 24 * 60 * 60 * 1000)  # Son 90 gün
        
        # Daha fazla veri alalım
        ohlcv = []
        current_start = start_time
        
        while current_start < end_time:
            print(f"Fetching data from {pd.to_datetime(current_start, unit='ms')}")
            
            batch = exchange.fetch_ohlcv(
                symbol='BTC/USDT',
                timeframe='1h',
                since=current_start,
                limit=1000
            )
            
            if not batch:
                break
                
            ohlcv.extend(batch)
            current_start = batch[-1][0] + 1  # Son mumun zamanından devam et
            
            # Her batch sonrası biraz bekle
            exchange.sleep(1000)  # 1 saniye bekle
        
        break  # Başarılı olduysa döngüden çık
        
    except Exception as e:
        retry_count += 1
        print(f"Error occurred: {str(e)}")
        print(f"Retry {retry_count}/{max_retries}")
        exchange.sleep(5000)  # 5 saniye bekle ve tekrar dene
        
        if retry_count == max_retries:
            raise Exception("Maximum retries reached. Could not fetch data.")

# DataFrame'e çevir
data = pd.DataFrame(
    ohlcv,
    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
)
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

# Optimizasyonu çalıştır
optimizer = StrategyOptimizer(
    strategy_class=YourStrategy,
    data=data,
    timeframe="1h"
)

print("\nStarting optimization...")
optimization_results = optimizer.optimize_strategy(n_trials=500)  # 500 deneme yapalım

# En iyi 5 parametre kombinasyonunu göster
print("\nTop 5 Parameter Combinations:")
for i, trial in enumerate(optimization_results['top_trials'], 1):
    print(f"\nRank {i}:")
    print(f"Parameters: {trial['params']}")
    print(f"Return: {trial['value']:.2%}")
    print(f"Duration: {trial['duration'].total_seconds():.2f}s")

print("\nOptimization completed!")

# Raporu oluştur
print("\nGenerating report...")
report_generator = ReportGenerator(
    strategy_name="Your Strategy",
    backtest_results=optimizer.backtest_results,
    optimization_results=optimization_results
)

report_path = report_generator.generate_report()
print(f"\nReport generated successfully: {report_path}") 
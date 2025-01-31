import sys
from pathlib import Path
from datetime import datetime, timedelta

# Proje kök dizinini Python path'ine ekle
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data_management import DataManager
from src.optimization import StrategyOptimizer
from src.reporting import ReportGenerator
from strategies.custom_strategies.your_strategy import YourStrategy

def run_optimization(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: datetime = None,
    end_date: datetime = None,
    n_trials: int = 200
) -> str:
    """
    Strateji optimizasyonu ve backtest yap
    
    Args:
        symbol: İşlem çifti (örn: "BTC/USDT")
        timeframe: Zaman dilimi (örn: "1h", "4h", "1d")
        start_date: Başlangıç tarihi
        end_date: Bitiş tarihi
        n_trials: Optimizasyon deneme sayısı
        
    Returns:
        str: Oluşturulan rapor dosyasının yolu
    """
    # Varsayılan tarihler
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=90)  # Son 90 gün
        
    print(f"Fetching data for {symbol}...")
    print(f"Period: {start_date} to {end_date}")
    print(f"Timeframe: {timeframe}")
    
    # Veri çek
    dm = DataManager(exchange='binance')
    data = dm.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    # Optimize et
    optimizer = StrategyOptimizer(
        strategy_class=YourStrategy,
        data=data,
        timeframe=timeframe
    )
    
    print(f"\nStarting optimization with {n_trials} trials...")
    optimization_results = optimizer.optimize(n_trials=n_trials)
    
    # Rapor oluştur
    report_generator = ReportGenerator(
        strategy_name=f"MA Cross Strategy ({symbol})",
        backtest_results=optimizer.backtest_results,
        optimization_results=optimization_results
    )
    
    report_path = report_generator.generate_report()
    print(f"\nReport generated: {report_path}")
    return report_path

if __name__ == "__main__":
    # 1 Kasım 2023'ten bugüne kadar
    end_date = datetime.now()
    start_date = datetime(2024,9,1)

    
    report_path = run_optimization(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date=start_date,
        end_date=end_date,
        n_trials=200  # Deneme sayısını 200'e çıkaralım
    ) 
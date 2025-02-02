import sys
from pathlib import Path
from datetime import datetime, timedelta

# Proje kök dizinini Python path'ine ekle
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data_management import DataManager
from src.strategy_optimizer import StrategyOptimizer
from src.reporting import ReportGenerator
from src.strategy_registry import StrategyRegistry

# Stratejileri import et (bu önemli!)
from strategies.ma_cross_strategy import MACrossStrategy
from strategies.rsi_strategy import RSIStrategy

def run_optimization(
    strategy_name: str,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: datetime = None,
    end_date: datetime = None,
    n_trials: int = 100,
    exchange: str = "binance"
) -> str:
    """
    Strateji optimizasyonu ve backtest yap
    
    Args:
        strategy_name: Kullanılacak strateji adı
        symbol: İşlem çifti (örn: "BTC/USDT")
        timeframe: Zaman dilimi (örn: "1h", "4h", "1d")
        start_date: Başlangıç tarihi
        end_date: Bitiş tarihi
        n_trials: Optimizasyon deneme sayısı
        exchange: Borsa adı (örn: "binance", "kucoin")
    """
    # Varsayılan tarihler
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=90)
        
    print(f"Strategy: {strategy_name}")
    print(f"Fetching data for {symbol}...")
    print(f"Period: {start_date} to {end_date}")
    print(f"Timeframe: {timeframe}")
    
    # Veri çek
    dm = DataManager(exchange=exchange)
    data = dm.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    if data.empty:
        print(f"Hata: {symbol} için veri çekilemedi. Lütfen farklı bir tarih aralığı deneyin.")
        return None
    
    # Strateji sınıfını al
    strategy_class = StrategyRegistry.get_strategy(strategy_name)
    
    # Optimize et
    optimizer = StrategyOptimizer(
        strategy_class=strategy_class,
        data=data,
        timeframe=timeframe
    )
    
    print(f"\nStarting optimization with {n_trials} trials...")
    optimization_results = optimizer.optimize(n_trials=n_trials)
    
    # Rapor oluştur
    report_generator = ReportGenerator(
        strategy_name=f"{strategy_name} ({symbol})",
        backtest_results=optimizer.backtest_results,
        optimization_results=optimization_results
    )
    
    report_path = report_generator.generate_report()
    print(f"\nReport generated: {report_path}")
    return report_path

if __name__ == "__main__":
    # Mevcut stratejileri listele
    available_strategies = StrategyRegistry.list_strategies()
    print("\nMevcut Stratejiler:")
    for i, strategy in enumerate(available_strategies, 1):
        print(f"{i}. {strategy}")
    
    # Kullanıcıdan strateji seçmesini iste
    while True:
        try:
            choice = input("\nHangi stratejiyi test etmek istersiniz? (numara girin, hepsini test etmek için 'all' yazın): ")
            
            if choice.lower() == 'all':
                strategies_to_test = available_strategies
                break
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(available_strategies):
                    strategies_to_test = [available_strategies[idx]]
                    break
                else:
                    print("Geçersiz numara! Lütfen tekrar deneyin.")
        except ValueError:
            print("Geçersiz giriş! Lütfen bir numara veya 'all' girin.")
    
    # Test parametrelerini al
    exchange = input("\nBorsa (binance, kucoin) [varsayılan: binance]: ") or "binance"
    timeframe = input("Zaman dilimi (1h, 4h, 1d) [varsayılan: 1h]: ") or "1h"
    start_date_str = input("Başlangıç tarihi (YYYY-MM-DD) [varsayılan: 1 yıl önce]: ")

    # Bitiş tarihi
    end_date_str = input("Bitiş tarihi (YYYY-MM-DD) [varsayılan: şimdi]: ")

    # Optimizasyon deneme sayısı
    n_trials_str = input("Optimizasyon deneme sayısı [varsayılan: 100]: ")
    n_trials = int(n_trials_str) if n_trials_str.isdigit() else 100

    # Tarihleri ayarla
    end_date = datetime.now()
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            print("Geçersiz bitiş tarihi formatı! Şimdiki zaman kullanılacak.")

    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            # 2017'den önceki tarihleri kontrol et
            min_date = datetime(2017, 1, 1)
            if start_date < min_date:
                print(f"Uyarı: {exchange} için {start_date.strftime('%Y-%m-%d')} tarihinden önceki veriler mevcut değil.")
                print(f"Başlangıç tarihi {min_date.strftime('%Y-%m-%d')} olarak ayarlandı.")
                start_date = min_date
        except ValueError:
            print("Geçersiz başlangıç tarihi formatı! Son 90 gün kullanılacak.")
            start_date = end_date - timedelta(days=90)
    else:
        start_date = datetime.now() - timedelta(days=365)  # 1 yıllık veri
    
    # Seçilen stratejileri test et
    for strategy_name in strategies_to_test:
        print(f"\nTesting {strategy_name}...")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        report_path = run_optimization(
            strategy_name=strategy_name,
            symbol="BTC/USDT",
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            n_trials=n_trials,
            exchange=exchange
        )
        print(f"Report generated: {report_path}\n") 
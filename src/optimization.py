import optuna
from typing import Dict, Any, Callable, List
import logging
from datetime import datetime
import pandas as pd
from src.strategy_management import StrategyParameters

class StrategyOptimizer:
    def __init__(self, 
                 strategy_class,
                 data: pd.DataFrame,
                 timeframe: str = "1h"):
        """
        Strateji optimizasyonu için ana sınıf
        
        Args:
            strategy_class: Optimize edilecek strateji sınıfı
            data: Kullanılacak fiyat verileri
            timeframe: Zaman dilimi
        """
        self.strategy_class = strategy_class
        self.data = data
        self.timeframe = timeframe
        self.logger = logging.getLogger(__name__)
        
        # Arama uzayını genişlet
        self.search_space = {
            'fast_ma': {
                'min': 3,
                'max': 100,
                'step': 1,     # 1'e düşürdük
                'type': 'int'
            },
            'slow_ma': {
                'min': 10,
                'max': 300,
                'step': 1,     # 2'den 1'e düşürdük
                'type': 'int'
            }
        }
        
    def objective(self, trial: optuna.Trial) -> float:
        """Bayesian optimizasyon için hedef fonksiyon"""
        params = {}
        
        # Her parametre için tip ve adım büyüklüğüne göre değer öner
        for param_name, param_config in self.search_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    step=param_config['step']
                )
            else:  # float
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    step=param_config['step']
                )
        
        # Fast MA, Slow MA'dan küçük olmalı
        if 'fast_ma' in params and 'slow_ma' in params:
            if params['fast_ma'] >= params['slow_ma']:
                return float('-inf')
        
        # StrategyParameters kullanarak parametreleri oluştur
        strategy_params = StrategyParameters(
            name="optimization_test",
            timeframe=self.timeframe,
            parameters=params
        )
        
        # Stratejiyi test et
        strategy = self.strategy_class(parameters=strategy_params)
        trades = strategy.execute_trades(self.data)
        performance = strategy.calculate_performance()
        
        # Her denemeyi detaylı logla
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        self.logger.info(f"Testing parameters: {param_str}")
        self.logger.info(f"Total return: {performance.get('total_return', 0):.2%}")
        self.logger.info(f"Win rate: {performance.get('win_rate', 0):.2%}")
        self.logger.info("-" * 50)
        
        return performance.get('total_return', float('-inf'))
    
    def optimize(self, n_trials: int = 200) -> Dict:
        """Bayesian optimizasyonu başlat"""
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=20,
                    seed=42
                )
            )
            
            self.logger.info(f"Starting Bayesian optimization with {n_trials} trials...")
            
            # İlerleme takibi için callback
            def callback(study, trial):
                if trial.number % 10 == 0:
                    self.logger.info(f"Trial {trial.number}/{n_trials}")
                    self.logger.info(f"Best value so far: {study.best_value:.2%}")
                    self.logger.info("-" * 50)
            
            study.optimize(self.objective, n_trials=n_trials, callbacks=[callback])
            
            # En iyi parametrelerle son bir backtest yap
            best_params = {
                'name': 'optimized_strategy',
                'timeframe': self.timeframe,
                'parameters': study.best_params
            }
            
            strategy = self.strategy_class(parameters=StrategyParameters(**best_params))
            trades = strategy.execute_trades(self.data)
            performance = strategy.calculate_performance()
            
            # Backtest sonuçlarını sakla
            self.backtest_results = {
                'trade_history': pd.DataFrame(trades),
                'performance_metrics': performance,
                'equity_curve': self._calculate_equity_curve(trades),
                'price_data': self.data
            }
            
            return {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'top_trials': self._get_diverse_trials(study),
                'optimization_history': study.trials_dataframe().to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
            raise

    def optimize_strategy(self, n_trials: int = 50) -> Dict:
        """Strateji parametrelerini optimize et"""
        try:
            # TPE sampler'ı daha fazla çeşitlilik için ayarla
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=10,     # 50'den 10'a düşürdük
                n_ei_candidates=20,      # 100'den 20'ye düşürdük
                seed=42,
                multivariate=True,
                constant_liar=True
            )
            
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler
            )
            
            # İlerleme takibi için callback
            def callback(study, trial):
                if trial.number % 10 == 0:
                    self.logger.info(f"Trial {trial.number}/{n_trials}")
                    self.logger.info(f"Best value so far: {study.best_value:.2%}")
                    self.logger.info("-" * 50)
            
            study.optimize(self.objective, n_trials=n_trials, callbacks=[callback])
            
            # En iyi parametrelerle son bir backtest yap
            best_params = {
                'name': 'optimized_strategy',
                'timeframe': self.timeframe,
                'parameters': study.best_params
            }
            
            strategy = self.strategy_class(parameters=StrategyParameters(**best_params))
            trades = strategy.execute_trades(self.data)
            performance = strategy.calculate_performance()
            
            # En iyi 5 kombinasyonu sakla - çeşitliliği artır
            all_trials = study.trials_dataframe()
            
            # Benzer parametreleri filtrele
            unique_params = set()
            diverse_trials = []
            
            for _, trial in all_trials.sort_values('value', ascending=False).iterrows():
                params = (
                    round(trial['params_fast_ma']), 
                    round(trial['params_slow_ma'])
                )
                
                if params not in unique_params and len(diverse_trials) < 5:
                    unique_params.add(params)
                    diverse_trials.append({
                        'params': {
                            'fast_ma': int(round(trial['params_fast_ma'])),
                            'slow_ma': int(round(trial['params_slow_ma']))
                        },
                        'value': trial['value'],
                        'duration': trial['duration']
                    })
            
            # Backtest sonuçlarını sakla
            self.backtest_results = {
                'trade_history': pd.DataFrame(trades),
                'performance_metrics': performance,
                'equity_curve': self._calculate_equity_curve(trades),
                'price_data': self.data
            }
            
            return {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'top_trials': diverse_trials,
                'optimization_history': study.trials_dataframe().to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
            raise

    def _calculate_equity_curve(self, trades: List[Dict]) -> pd.Series:
        """İşlemlerden equity eğrisi oluştur"""
        if not trades:
            return pd.Series()
        
        # İşlemleri DataFrame'e çevir
        trades_df = pd.DataFrame(trades)
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Kümülatif return hesapla
        equity = (1 + trades_df['pnl']).cumprod()
        equity.index = trades_df['exit_time']
        
        return equity 

    def _get_diverse_trials(self, study: optuna.Study) -> List[Dict]:
        """En iyi ve çeşitli denemeleri getir"""
        trials_df = study.trials_dataframe()
        if trials_df.empty:
            return []
        
        # En iyi denemeleri al
        top_trials = []
        used_params = set()  # Kullanılan parametre kombinasyonlarını takip et
        
        for _, trial in trials_df.sort_values('value', ascending=False).iterrows():
            fast_ma = int(round(trial['params_fast_ma']))
            slow_ma = int(round(trial['params_slow_ma']))
            
            # Bu parametre kombinasyonu daha önce kullanıldı mı?
            param_key = (fast_ma, slow_ma)
            if param_key not in used_params and len(top_trials) < 5:
                # Stratejiyi bu parametrelerle test et
                strategy_params = StrategyParameters(
                    name="test_strategy",
                    timeframe=self.timeframe,
                    parameters={'fast_ma': fast_ma, 'slow_ma': slow_ma}
                )
                strategy = self.strategy_class(parameters=strategy_params)
                trades = strategy.execute_trades(self.data)
                
                # İşlem istatistiklerini hesapla
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t['pnl'] > 0])
                losing_trades = total_trades - winning_trades
                win_rate = (winning_trades/total_trades*100) if total_trades > 0 else 0
                
                top_trials.append({
                    'params': {
                        'fast_ma': fast_ma,
                        'slow_ma': slow_ma
                    },
                    'value': float(trial['value']),
                    'trade_stats': {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'losing_trades': losing_trades,
                        'win_rate': f"{win_rate:.1f}%"
                    }
                })
                used_params.add(param_key)
        
        return top_trials 
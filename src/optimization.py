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
                'max': 100,    # 50'den 100'e çıkardık
                'step': 1,
                'type': 'int'
            },
            'slow_ma': {
                'min': 10,
                'max': 300,    # 200'den 300'e çıkardık
                'step': 2,
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
    
    def optimize(self) -> Dict[str, Any]:
        """Bayesian optimizasyonu başlat"""
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=10,
                    seed=42
                )
            )
            
            n_trials = 50
            self.logger.info(f"Starting Bayesian optimization with {n_trials} trials...")
            
            # İlerleme takibi için callback
            def callback(study, trial):
                if trial.number % 5 == 0:
                    self.logger.info(f"Trial {trial.number}/{n_trials}")
                    self.logger.info(f"Best value so far: {study.best_value:.2%}")
                    self.logger.info("-" * 50)
            
            study.optimize(self.objective, n_trials=n_trials, callbacks=[callback])
            
            # Parametreleri tam sayıya yuvarla
            best_params = {
                'fast_ma': int(round(study.best_params['fast_ma'])),
                'slow_ma': int(round(study.best_params['slow_ma']))
            }
            best_value = study.best_value
            
            self.logger.info(f"Bayesian optimization completed.")
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Best total return: {best_value:.2%}")
            
            # En iyi parametrelerin etrafındaki bölgeyi incele
            self.logger.info("\nTop 5 parameter combinations:")
            trials_df = study.trials_dataframe()
            top_trials = trials_df.nlargest(5, 'value')
            for _, trial in top_trials.iterrows():
                self.logger.info(
                    f"Fast MA: {int(round(trial['params_fast_ma']))}, "
                    f"Slow MA: {int(round(trial['params_slow_ma']))}, "
                    f"Return: {trial['value']:.2%}"
                )
            
            # Top trials'ı da tam sayıya yuvarla
            top_trials_rounded = []
            for trial in top_trials.to_dict('records'):
                top_trials_rounded.append({
                    'params_fast_ma': int(round(trial['params_fast_ma'])),
                    'params_slow_ma': int(round(trial['params_slow_ma'])),
                    'value': trial['value']
                })
            
            return {
                'best_params': best_params,
                'best_value': best_value,
                'study': study,
                'top_trials': top_trials_rounded
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
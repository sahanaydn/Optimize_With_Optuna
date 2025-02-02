from typing import Dict, Any, Type
import optuna
import pandas as pd
from src.strategy_base import BaseStrategy, StrategyParameters
import numpy as np

class StrategyOptimizer:
    def __init__(self, 
                 strategy_class: Type[BaseStrategy],
                 data: pd.DataFrame,
                 timeframe: str = "1h"):
        self.strategy_class = strategy_class
        self.data = data
        self.timeframe = timeframe
        self.parameter_space = strategy_class({}).parameter_space
        
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """Bayesian optimizasyon yap"""
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self._objective(trial, self.data),
            n_trials=n_trials
        )
        
        # En iyi parametrelerle son bir backtest yap
        best_params = self._suggest_parameters(study.best_trial)
        strategy = self.strategy_class(best_params)
        trades = strategy.execute_trades(self.data)
        performance = strategy.calculate_performance()
        
        # Equity curve hesapla
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            equity = (1 + trades_df['pnl']).cumprod()
            equity.index = trades_df['exit_time']
        else:
            equity = pd.Series([1.0])
        
        # Backtest sonuçlarını sakla
        self.backtest_results = {
            'trade_history': trades_df,
            'performance_metrics': performance,
            'price_data': self.data,
            'equity_curve': equity
        }
        
        return self._process_results(study)
        
    def _objective(self, trial: optuna.Trial, data: pd.DataFrame) -> float:
        params = self._suggest_parameters(trial)
        
        if not self.strategy_class(params).validate_parameters(params):
            return float('-inf')
        
        strategy = self.strategy_class(params)
        trades = strategy.execute_trades(data)
        metrics = strategy.calculate_performance()
        
        if metrics['total_trades'] < 10:
            return float('-inf')
        
        return metrics['total_return']  # Sadece total return'e göre optimize et
        
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Parametre önerisi oluştur"""
        params = {}
        for name, config in self.parameter_space.items():
            if config['type'] == 'categorical':
                params[name] = trial.suggest_categorical(
                    name=name,
                    choices=config['options']
                )
            elif config['type'] == 'int':
                params[name] = trial.suggest_int(
                    name=name,
                    low=config['min'],
                    high=config['max'],
                    step=config['step']
                )
            else:
                params[name] = trial.suggest_float(
                    name=name,
                    low=config['min'],
                    high=config['max'],
                    step=config['step']
                )
        return params 

    def _process_results(self, study: optuna.Study) -> Dict[str, Any]:
        """Optimizasyon sonuçlarını işle"""
        # En iyi parametrelerle son bir backtest yap
        best_params = self._suggest_parameters(study.best_trial)
        strategy = self.strategy_class(best_params)
        trades = strategy.execute_trades(self.data)
        performance = strategy.calculate_performance()
        
        # Equity curve hesapla
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            equity = (1 + trades_df['pnl']).cumprod()
            equity.index = trades_df['exit_time']
        else:
            equity = pd.Series([1.0])
        
        # Backtest sonuçlarını sakla
        self.backtest_results = {
            'trade_history': trades_df,
            'performance_metrics': performance,
            'price_data': self.data,
            'equity_curve': equity
        }
        
        # En iyi 5 farklı denemeyi al
        trials_df = study.trials_dataframe()
        used_params = set()
        top_trials = []
        
        # Tüm denemeleri performansa göre sırala
        for _, trial in trials_df.sort_values('value', ascending=False).iterrows():
            params = {}
            for name in self.parameter_space.keys():
                param_value = trial[f'params_{name}']
                if self.parameter_space[name]['type'] == 'int':
                    param_value = int(round(param_value))
                params[name] = param_value
            
            param_key = tuple(sorted(params.items()))
            if param_key not in used_params and len(top_trials) < 5:
                # Bu parametrelerle test yap
                strategy = self.strategy_class(params)
                test_trades = strategy.execute_trades(self.data)
                test_metrics = strategy.calculate_performance()
                total_trades = len(test_trades)
                winning_trades = len([t for t in test_trades if t['pnl'] > 0])
                
                top_trials.append({
                    'params': params,
                    'value': test_metrics['total_return'],
                    'trade_stats': {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'losing_trades': total_trades - winning_trades,
                        'win_rate': f"{(winning_trades/total_trades*100):.1f}%" if total_trades > 0 else "0%"
                    }
                })
                used_params.add(param_key)
        
        return {
            'best_params': study.best_params,
            'best_value': performance['total_return'],
            'top_trials': top_trials
        } 
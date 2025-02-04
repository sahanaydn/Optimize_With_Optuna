from typing import Dict, Any, Type, List
import optuna
import pandas as pd
from src.strategy_base import BaseStrategy, StrategyParameters
import numpy as np
import itertools

class StrategyOptimizer:
    def __init__(self, 
                 strategy_class: Type[BaseStrategy],
                 data: pd.DataFrame,
                 timeframe: str = "1h",
                 optimizer_type: str = "bayesian"):
        self.strategy_class = strategy_class
        self.data = data
        self.timeframe = timeframe
        self.parameter_space = strategy_class({}).parameter_space
        self.optimizer_type = optimizer_type
        
    def optimize(self, n_trials: int = None) -> Dict[str, Any]:
        """Seçilen optimizasyon algoritmasını çalıştır"""
        if self.optimizer_type == "grid":
            return self._grid_search()
        else:  # bayesian
            if n_trials is None:
                n_trials = 100
            return self._bayesian_optimization(n_trials)
        
    def _grid_search(self) -> Dict[str, Any]:
        """Grid Search ile tüm kombinasyonları dene"""
        best_value = float('-inf')
        best_params = None
        trials = []
        
        # Parameter space'den grid noktaları oluştur
        param_grid = {}
        for name, config in self.parameter_space.items():
            if config['type'] == 'int':
                param_grid[name] = range(
                    config['min'], 
                    config['max'] + 1, 
                    config.get('step', 1)
                )
                
        # Tüm kombinasyonları dene
        for params in self._generate_combinations(param_grid):
            value = self._objective_func(params)
            trials.append({
                'params': params,
                'value': value
            })
            
            if value > best_value:
                best_value = value
                best_params = params
        
        # En iyi parametrelerle son bir backtest yap
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
            'equity_curve': equity,
            'timeframe': self.timeframe
        }
            
        return {
            'best_params': best_params,
            'best_value': best_value,
            'trials': trials
        }
        
    def _bayesian_optimization(self, n_trials: int) -> Dict[str, Any]:
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
            'equity_curve': equity,
            'timeframe': self.timeframe
        }
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'top_trials': self._get_diverse_trials(study)
        }
        
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

    def _get_diverse_trials(self, study: optuna.Study) -> List[Dict[str, Any]]:
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
        
        return top_trials 

    def _generate_combinations(self, param_grid: Dict) -> List[Dict]:
        """Parameter grid'inden tüm kombinasyonları oluştur"""
        # Her parametrenin değerlerini al
        param_values = [param_grid[param] for param in param_grid]
        
        # Tüm kombinasyonları oluştur
        combinations = []
        for values in itertools.product(*param_values):
            params = dict(zip(param_grid.keys(), values))
            
            # Parametre validasyonu
            if self.strategy_class(params).validate_parameters(params):
                combinations.append(params)
            
        return combinations

    def _objective_func(self, params: Dict) -> float:
        """Grid search için objective fonksiyon"""
        strategy = self.strategy_class(params)
        trades = strategy.execute_trades(self.data)
        metrics = strategy.calculate_performance()
        
        if metrics['total_trades'] < 10:
            return float('-inf')
        
        return metrics['total_return'] 
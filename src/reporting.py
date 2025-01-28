from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path
import jinja2
import pdfkit
import json
import optuna
import optuna.visualization.matplotlib
import base64
import io
import numpy as np
from jinja2 import Template

class ReportGenerator:
    def __init__(self, 
                 strategy_name: str,
                 backtest_results: Dict[str, Any],
                 optimization_results: Dict[str, Any] = None):
        """
        Rapor oluşturucu sınıf
        """
        self.strategy_name = strategy_name
        self.backtest_results = backtest_results
        self.optimization_results = optimization_results
        self.logger = logging.getLogger(__name__)
        
        # Sadece PDF raporu için klasör oluştur
        self.report_dir = Path('reports')
        self.report_dir.mkdir(exist_ok=True)
        
        # PDF dosya yolu
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_path = self.report_dir / f"report_{timestamp}.pdf"
        
    def generate_report(self) -> str:
        """PDF rapor oluştur"""
        try:
            # Buy & Hold return hesapla
            first_price = self.backtest_results['price_data']['close'].iloc[0]
            last_price = self.backtest_results['price_data']['close'].iloc[-1]
            buy_hold_return = ((last_price - first_price) / first_price)  # Ondalık olarak

            # Test edilen tarih aralığını hesapla
            trades_df = self.backtest_results['trade_history']
            test_start = pd.to_datetime(trades_df['entry_time'].min()).strftime("%Y-%m-%d %H:%M")
            test_end = pd.to_datetime(trades_df['exit_time'].max()).strftime("%Y-%m-%d %H:%M")
            timeframe = self.backtest_results.get('timeframe', '1h')
            
            # Rapor bileşenlerini hazırla
            performance_summary = self._create_performance_summary()
            performance_summary['Buy & Hold Return'] = buy_hold_return  # Buy & Hold'u ekle
            trade_analysis = self._create_trade_analysis()
            optimization_results = self._create_optimization_summary() if self.optimization_results else None
            comparison_chart = self._create_comparison_chart()
            
            # HTML şablonunu yükle ve içeriği oluştur
            template = self._load_template()
            html_content = template.render(
                strategy_name=self.strategy_name,
                generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                test_period={
                    'start': test_start,
                    'end': test_end,
                    'timeframe': timeframe
                },
                performance_summary=performance_summary,
                trade_analysis=trade_analysis,
                optimization_results=optimization_results,
                comparison_chart=comparison_chart
            )
            
            # PDF oluştur
            options = {
                'quiet': '',
                'enable-local-file-access': None
            }
            
            pdfkit.from_string(
                html_content, 
                str(self.report_path),
                options=options,
                css=str(Path(__file__).parent / 'report_style.css')
            )
            
            self.logger.info(f"Report generated: {self.report_path}")
            return str(self.report_path)
            
        except Exception as e:
            self.logger.error(f"Report generation error: {str(e)}")
            raise
            
    def _create_trade_analysis(self) -> Dict:
        """İşlem analizlerini yap"""
        trades_df = self.backtest_results['trade_history']
        
        # Trade listesini oluştur
        trade_list = []
        for _, trade in trades_df.iterrows():
            pnl_value = trade['pnl'] * 100
            pnl_str = f"+{pnl_value:.2f}%" if pnl_value > 0 else f"{pnl_value:.2f}%"
            
            duration = pd.to_datetime(trade['exit_time']) - pd.to_datetime(trade['entry_time'])
            duration_str = self._format_duration(duration)
            
            trade_list.append({
                'Entry Time': pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M'),
                'Exit Time': pd.to_datetime(trade['exit_time']).strftime('%Y-%m-%d %H:%M'),
                'Type': trade['type'].upper(),
                'Entry Price': f"${trade['entry_price']:.2f}",
                'Exit Price': f"${trade['exit_price']:.2f}",
                'PnL': pnl_str,
                'Duration': duration_str,
                'Result': 'WIN' if trade['pnl'] > 0 else 'LOSS'
            })
        
        # Trade istatistiklerini hesapla
        total_trades = len(trades_df)
        long_trades = len(trades_df[trades_df['type'] == 'long'])
        short_trades = len(trades_df[trades_df['type'] == 'short'])
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        
        # Trade duration istatistikleri
        duration_stats = self._calculate_trade_duration_stats(trades_df)
        
        return {
            'Statistics': {  # İstatistikleri ayrı bir sözlükte topla
                'Total Trades': total_trades,
                'Long Trades': long_trades,
                'Short Trades': short_trades,
                'Winning Trades': winning_trades,
                'Losing Trades': total_trades - winning_trades,
                'Average Duration': duration_stats['mean'],
                'Longest Trade': duration_stats['max'],
                'Shortest Trade': duration_stats['min']
            },
            'Trades': trade_list  # Trade listesi ayrı
        }
        
    def _create_performance_summary(self) -> Dict:
        """Performans özeti oluştur"""
        metrics = self.backtest_results['performance_metrics']
        
        return {
            'Total Return': metrics['total_return'],
            'Number of Trades': metrics['total_trades'],
            'Win Rate': metrics['win_rate'],
            'Profit Factor': metrics['profit_factor'],
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Max Drawdown': metrics['max_drawdown'],
            'Sortino Ratio': metrics['sortino_ratio'],
            'Calmar Ratio': metrics['calmar_ratio']
        }
        
    def _calculate_trade_duration_stats(self, trades_df: pd.DataFrame) -> Dict:
        """İşlem süresi istatistiklerini hesapla"""
        if trades_df.empty:
            return {
                'min': '0h',
                'max': '0h',
                'mean': '0h',
                'median': '0h'
            }
            
        trades_df['duration'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
        
        return {
            'min': self._format_duration(trades_df['duration'].min()),
            'max': self._format_duration(trades_df['duration'].max()),
            'mean': self._format_duration(trades_df['duration'].mean()),
            'median': self._format_duration(trades_df['duration'].median())
        }
        
    def _format_duration(self, duration) -> str:
        """Süreyi okunabilir formata çevir"""
        total_seconds = duration.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        
        if hours == 0:
            return f"{minutes}m"
        elif minutes == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {minutes}m"
        
    def _create_optimization_summary(self) -> Dict:
        """Optimizasyon özeti oluştur"""
        if not self.optimization_results:
            return None
            
        # En iyi 5 kombinasyonu da ekle
        top_trials = self.optimization_results.get('top_trials', [])
        
        return {
            'best_params': self.optimization_results['best_params'],
            'best_value': self.optimization_results['best_value'],
            'top_trials': top_trials
        }
        
    def _create_comparison_chart(self) -> str:
        """Buy & Hold vs Strateji karşılaştırma grafiği oluştur"""
        try:
            self.logger.info("Creating comparison chart...")
            
            # Equity eğrilerini al
            equity_curve = self.backtest_results['equity_curve']
            trades_df = self.backtest_results['trade_history']
            
            # Fiyat serisini al
            price_data = self.backtest_results.get('price_data')  # Bunu backtester'dan almamız gerekecek
            if price_data is None:
                # Eğer price_data yoksa, trade'lerden yaklaşık bir fiyat serisi oluştur
                dates = pd.date_range(start=equity_curve.index[0], 
                                    end=equity_curve.index[-1], 
                                    freq='1H')  # Saatlik veri
                
                # İlk ve son fiyatları kullan
                initial_price = trades_df['entry_price'].iloc[0]
                final_price = trades_df['exit_price'].iloc[-1]
                
                # Lineer interpolasyon ile fiyat serisi oluştur
                price_series = pd.Series(
                    np.linspace(initial_price, final_price, len(dates)),
                    index=dates
                )
            else:
                price_series = price_data['close']
            
            # Buy & Hold equity eğrisini hesapla
            buy_hold_equity = (price_series / price_series.iloc[0])
            
            # Grafik oluştur
            plt.figure(figsize=(12, 6))
            
            # Strateji performansı
            plt.plot(equity_curve.index, equity_curve.values, 
                    label='Strategy', color='#2ecc71', linewidth=2)
            
            # Buy & Hold performansı
            plt.plot(buy_hold_equity.index, buy_hold_equity.values, 
                    label='Buy & Hold', color='#3498db', linewidth=2, linestyle='--')
            
            plt.title('Strategy vs Buy & Hold Performance')
            plt.xlabel('Date')
            plt.ylabel('Return (1 = 100%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Tarihleri düzgün formatlama
            plt.gcf().autofmt_xdate()
            
            # Grafiği base64'e çevir
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            # Base64 string oluştur
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            self.logger.info("Chart created successfully!")
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            self.logger.error(f"Chart generation error: {str(e)}")
            self.logger.exception(e)
            return None
        
    def _load_template(self) -> jinja2.Template:
        """HTML şablonunu yükle"""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ strategy_name }} - Strategy Report</title>
            <meta charset="UTF-8">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }
                .trade-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                .trade-table th, .trade-table td {
                    padding: 8px;
                    text-align: left;
                    border: 1px solid #ddd;
                }
                .trade-table th {
                    background-color: #f5f5f5;
                }
                .trade-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .profit {
                    color: #27ae60;
                    font-weight: bold;
                }
                .loss {
                    color: #e74c3c;
                    font-weight: bold;
                }
                .win-badge {
                    color: #27ae60;
                    font-weight: bold;
                }
                .loss-badge {
                    color: #e74c3c;
                    font-weight: bold;
                }
                .report-header {
                    margin-bottom: 30px;
                }
                .report-header p {
                    color: #666;
                    margin: 5px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="report-header">
                    <h1>{{ strategy_name }} - Strategy Report</h1>
                    <p>Report Generated: {{ generation_date }}</p>
                    <p>Test Period: {{ test_period.start }} to {{ test_period.end }}</p>
                    <p>Timeframe: {{ test_period.timeframe }}</p>
                </div>
                
                {% if performance_summary %}
                <div class="section">
                    <h2>Performance Summary</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        {% for metric, value in performance_summary.items() %}
                        <tr>
                            <th>{{ metric }}</th>
                            <td>
                                {% if 'return' in metric.lower() or 'rate' in metric.lower() %}
                                    {{ "%.2f%%"|format(value * 100) }}
                                {% else %}
                                    {{ value }}
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}
                
                {% if optimization_results %}
                <div class="section">
                    <h2>Optimization Results</h2>
                    
                    <h3>Search Space</h3>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Range</th>
                            <th>Step Size</th>
                        </tr>
                        <tr>
                            <th>Fast MA</th>
                            <td>5 to 30</td>
                            <td>1</td>
                        </tr>
                        <tr>
                            <th>Slow MA</th>
                            <td>10 to 100</td>
                            <td>1</td>
                        </tr>
                    </table>
                    
                    <h3>Best Parameters</h3>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                        {% for param, value in optimization_results.best_params.items() %}
                        <tr>
                            <th>{{ param }}</th>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                        <tr>
                            <th>Best Return</th>
                            <td>{{ "%.2f%%"|format(optimization_results.best_value * 100) }}</td>
                        </tr>
                    </table>
                    
                    <h3>Top 5 Parameter Combinations</h3>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Parameters</th>
                            <th>Return</th>
                        </tr>
                        {% for trial in optimization_results.top_trials %}
                        <tr>
                            <td>{{ loop.index }}</td>
                            <td>
                                {% for param, value in trial.params.items() %}
                                    {{ param }}: {{ value }}{% if not loop.last %}, {% endif %}
                                {% endfor %}
                            </td>
                            <td>{{ "%.2f%%"|format(trial.value * 100) }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}
                
                {% if trade_analysis %}
                <div class="section">
                    <h2>Trade Statistics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        {% for metric, value in trade_analysis.Statistics.items() %}
                        <tr>
                            <th>{{ metric }}</th>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Trade List</h2>
                    <table class="trade-table">
                        <thead>
                            <tr>
                                <th>Entry Time</th>
                                <th>Exit Time</th>
                                <th>Type</th>
                                <th>Entry Price</th>
                                <th>Exit Price</th>
                                <th>PnL</th>
                                <th>Duration</th>
                                <th>Result</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for trade in trade_analysis.Trades %}
                            <tr>
                                <td>{{ trade['Entry Time'] }}</td>
                                <td>{{ trade['Exit Time'] }}</td>
                                <td>{{ trade.Type }}</td>
                                <td>{{ trade['Entry Price'] }}</td>
                                <td>{{ trade['Exit Price'] }}</td>
                                <td class="{{ 'profit' if '+' in trade.PnL else 'loss' }}">
                                    {{ trade.PnL }}
                                </td>
                                <td>{{ trade.Duration }}</td>
                                <td class="{{ 'win-badge' if 'WIN' in trade.Result else 'loss-badge' }}">
                                    {{ '✓ WIN' if 'WIN' in trade.Result else '✗ LOSS' }}
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% endif %}
                
                {% if comparison_chart %}
                <div class="section">
                    <h2>Strategy Comparison</h2>
                    <img src="{{ comparison_chart }}" alt="Strategy vs Buy & Hold Comparison">
                </div>
                {% endif %}
            </div>
        </body>
        </html>
        """
        return jinja2.Template(template_str) 
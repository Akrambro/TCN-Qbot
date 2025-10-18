"""
Logging and Monitoring Utilities

Provides logging functionality for trading bot operations
"""

import logging
import os
from datetime import datetime
from typing import Dict, Optional
import json


class TradingLogger:
    """
    Structured logger for trading operations
    """
    
    def __init__(
        self,
        log_dir: str = 'logs',
        log_level: int = logging.INFO,
        console_output: bool = True
    ):
        """
        Initialize trading logger
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console_output: Whether to also print to console
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('TCNTradingBot')
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler (daily rotation)
        log_filename = os.path.join(
            log_dir,
            f"trades_{datetime.now().strftime('%Y-%m-%d')}.log"
        )
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(log_level)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            self.logger.addHandler(console_handler)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Trade log file (JSON format for easy parsing)
        self.trade_log_file = os.path.join(
            log_dir,
            f"trades_detail_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )
    
    def log_startup(self, config: Dict):
        """Log bot startup with configuration"""
        self.logger.info("="*60)
        self.logger.info("TCN TRADING BOT STARTED")
        self.logger.info("="*60)
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def log_model_loaded(self, model_path: str, model_info: Dict):
        """Log model loading"""
        self.logger.info(f"Model loaded from: {model_path}")
        self.logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
    
    def log_data_fetch(self, asset: str, n_candles: int):
        """Log data fetching"""
        self.logger.debug(f"Fetched {n_candles} candles for {asset}")
    
    def log_prediction(
        self,
        timestamp: datetime,
        prediction_proba: float,
        confidence: float,
        features: Optional[Dict] = None
    ):
        """Log model prediction"""
        direction = "CALL" if prediction_proba >= 0.5 else "PUT"
        self.logger.info(
            f"Prediction at {timestamp}: {direction} "
            f"(probability: {prediction_proba:.4f}, confidence: {confidence:.4f})"
        )
        
        if features:
            self.logger.debug(f"Features: {json.dumps(features, indent=2)}")
    
    def log_trade_signal(
        self,
        timestamp: datetime,
        signal: str,
        prediction_proba: float,
        reason: str
    ):
        """Log trade signal decision"""
        self.logger.info(
            f"Signal at {timestamp}: {signal} "
            f"(probability: {prediction_proba:.4f}) - {reason}"
        )
    
    def log_trade_execution(self, trade: Dict):
        """
        Log trade execution
        
        Args:
            trade: Dictionary with trade details
        """
        self.logger.info(
            f"Trade executed: {trade['direction']} "
            f"Amount: ${trade['amount']:.2f} "
            f"Asset: {trade['asset']} "
            f"Expiry: {trade['expiry']}s"
        )
        
        # Append to JSON lines file
        with open(self.trade_log_file, 'a') as f:
            # Convert datetime objects to strings
            trade_copy = trade.copy()
            for key, value in trade_copy.items():
                if isinstance(value, datetime):
                    trade_copy[key] = value.isoformat()
            
            f.write(json.dumps(trade_copy) + '\n')
    
    def log_trade_result(
        self,
        trade_id: str,
        result: str,
        pnl: float,
        balance: float
    ):
        """Log trade result"""
        symbol = "✅" if result == "win" else "❌"
        self.logger.info(
            f"{symbol} Trade {trade_id} result: {result.upper()} "
            f"P&L: ${pnl:+.2f} "
            f"Balance: ${balance:.2f}"
        )
    
    def log_daily_summary(self, summary: Dict):
        """Log daily trading summary"""
        self.logger.info("="*60)
        self.logger.info("DAILY SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total trades: {summary['total_trades']}")
        self.logger.info(f"Wins: {summary['wins']} ({summary['win_rate']:.2%})")
        self.logger.info(f"Losses: {summary['losses']}")
        self.logger.info(f"Total P&L: ${summary['total_pnl']:+.2f}")
        self.logger.info(f"Win rate: {summary['win_rate']:.2%}")
        self.logger.info(f"Balance: ${summary['balance']:.2f}")
        self.logger.info("="*60)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        self.logger.error(
            f"Error in {context}: {type(error).__name__}: {str(error)}",
            exc_info=True
        )
    
    def log_warning(self, message: str):
        """Log warning"""
        self.logger.warning(message)
    
    def log_risk_alert(self, alert_type: str, details: Dict):
        """Log risk management alert"""
        self.logger.warning(
            f"RISK ALERT: {alert_type} - {json.dumps(details)}"
        )
    
    def log_shutdown(self, final_summary: Dict):
        """Log bot shutdown"""
        self.logger.info("="*60)
        self.logger.info("TCN TRADING BOT SHUTDOWN")
        self.logger.info("="*60)
        self.logger.info(f"Final summary: {json.dumps(final_summary, indent=2)}")


class PerformanceMonitor:
    """
    Monitor and track performance metrics
    """
    
    def __init__(self):
        self.metrics = {
            'predictions': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'prediction_time_ms': [],
            'feature_calc_time_ms': []
        }
    
    def record_prediction_time(self, time_ms: float):
        """Record prediction latency"""
        self.metrics['prediction_time_ms'].append(time_ms)
        self.metrics['predictions'] += 1
    
    def record_feature_calc_time(self, time_ms: float):
        """Record feature calculation time"""
        self.metrics['feature_calc_time_ms'].append(time_ms)
    
    def record_trade(self, is_win: bool, pnl: float):
        """Record trade result"""
        self.metrics['trades'] += 1
        if is_win:
            self.metrics['wins'] += 1
        else:
            self.metrics['losses'] += 1
        self.metrics['total_pnl'] += pnl
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        import numpy as np
        
        stats = {
            'total_predictions': self.metrics['predictions'],
            'total_trades': self.metrics['trades'],
            'wins': self.metrics['wins'],
            'losses': self.metrics['losses'],
            'win_rate': self.metrics['wins'] / max(1, self.metrics['trades']),
            'total_pnl': self.metrics['total_pnl']
        }
        
        if self.metrics['prediction_time_ms']:
            stats['avg_prediction_time_ms'] = np.mean(self.metrics['prediction_time_ms'])
            stats['max_prediction_time_ms'] = np.max(self.metrics['prediction_time_ms'])
        
        if self.metrics['feature_calc_time_ms']:
            stats['avg_feature_calc_time_ms'] = np.mean(self.metrics['feature_calc_time_ms'])
        
        return stats
    
    def print_statistics(self):
        """Print performance statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS")
        print("="*60)
        
        print(f"\nTrading:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Total trades: {stats['total_trades']}")
        print(f"  Wins: {stats['wins']} ({stats['win_rate']:.2%})")
        print(f"  Losses: {stats['losses']}")
        print(f"  Total P&L: ${stats['total_pnl']:+.2f}")
        
        if 'avg_prediction_time_ms' in stats:
            print(f"\nLatency:")
            print(f"  Avg prediction time: {stats['avg_prediction_time_ms']:.2f} ms")
            print(f"  Max prediction time: {stats['max_prediction_time_ms']:.2f} ms")
        
        if 'avg_feature_calc_time_ms' in stats:
            print(f"  Avg feature calc time: {stats['avg_feature_calc_time_ms']:.2f} ms")
        
        print("="*60)


# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = TradingLogger(log_dir='logs', console_output=True)
    
    # Log startup
    config = {
        'model': 'tcn_forex_model.pt',
        'asset': 'EURUSD',
        'min_confidence': 0.60,
        'trade_amount': 10.0
    }
    logger.log_startup(config)
    
    # Log model loading
    logger.log_model_loaded(
        'models/tcn_forex_model.pt',
        {'input_channels': 30, 'receptive_field': 15}
    )
    
    # Log prediction
    logger.log_prediction(
        timestamp=datetime.now(),
        prediction_proba=0.75,
        confidence=0.75
    )
    
    # Log trade execution
    trade = {
        'timestamp': datetime.now(),
        'direction': 'CALL',
        'amount': 10.0,
        'asset': 'EURUSD',
        'expiry': 60,
        'prediction_proba': 0.75
    }
    logger.log_trade_execution(trade)
    
    # Log trade result
    logger.log_trade_result(
        trade_id='trade_001',
        result='win',
        pnl=8.0,
        balance=1008.0
    )
    
    # Log daily summary
    summary = {
        'total_trades': 10,
        'wins': 6,
        'losses': 4,
        'win_rate': 0.60,
        'total_pnl': 8.0,
        'balance': 1008.0
    }
    logger.log_daily_summary(summary)
    
    # Performance monitor
    monitor = PerformanceMonitor()
    monitor.record_prediction_time(45.5)
    monitor.record_feature_calc_time(12.3)
    monitor.record_trade(True, 8.0)
    monitor.print_statistics()

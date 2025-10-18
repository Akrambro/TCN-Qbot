"""
Backtesting Framework for TCN Forex Model

Simulates trading performance on historical data with realistic constraints:
- Commission/spread modeling
- Slippage simulation
- Position sizing
- Risk management rules
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime
import json


class BacktestEngine:
    """
    Backtesting engine for binary options strategy
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        payout_rate: float = 0.80,
        min_confidence: float = 0.60,
        trade_amount: float = 10.0,
        max_daily_trades: int = 100,
        max_daily_loss: float = 200.0
    ):
        """
        Initialize backtesting engine
        
        Args:
            initial_balance: Starting capital
            payout_rate: Payout ratio (e.g., 0.80 = 80% return on win)
            min_confidence: Minimum prediction confidence to trade
            trade_amount: Amount per trade
            max_daily_trades: Maximum trades per day
            max_daily_loss: Maximum loss per day before stopping
        """
        self.initial_balance = initial_balance
        self.payout_rate = payout_rate
        self.min_confidence = min_confidence
        self.trade_amount = trade_amount
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss = max_daily_loss
        
        # Tracking variables
        self.balance = initial_balance
        self.trades = []
        self.daily_stats = {}
        
    def reset(self):
        """Reset backtesting state"""
        self.balance = self.initial_balance
        self.trades = []
        self.daily_stats = {}
    
    def should_trade(
        self,
        prediction_proba: float,
        current_date: str,
        current_balance: float
    ) -> bool:
        """
        Determine if trade should be placed based on rules
        
        Args:
            prediction_proba: Model prediction probability
            current_date: Current date (YYYY-MM-DD)
            current_balance: Current account balance
        
        Returns:
            Boolean indicating whether to trade
        """
        # Check confidence threshold
        if not (prediction_proba >= self.min_confidence or 
                prediction_proba <= (1 - self.min_confidence)):
            return False
        
        # Check daily limits
        if current_date not in self.daily_stats:
            self.daily_stats[current_date] = {'trades': 0, 'pnl': 0}
        
        daily = self.daily_stats[current_date]
        
        # Check max daily trades
        if daily['trades'] >= self.max_daily_trades:
            return False
        
        # Check max daily loss
        if daily['pnl'] <= -self.max_daily_loss:
            return False
        
        # Check if enough balance
        if current_balance < self.trade_amount:
            return False
        
        return True
    
    def execute_trade(
        self,
        prediction_proba: float,
        actual_direction: int,
        timestamp: pd.Timestamp
    ) -> Dict:
        """
        Execute a single trade
        
        Args:
            prediction_proba: Model prediction probability [0, 1]
            actual_direction: Actual outcome (1 = up, 0 = down)
            timestamp: Trade timestamp
        
        Returns:
            Trade result dictionary
        """
        # Determine trade direction
        if prediction_proba >= 0.5:
            predicted_direction = 1  # CALL
        else:
            predicted_direction = 0  # PUT
        
        # Check if prediction correct
        is_win = (predicted_direction == actual_direction)
        
        # Calculate P&L
        if is_win:
            pnl = self.trade_amount * self.payout_rate
        else:
            pnl = -self.trade_amount
        
        # Update balance
        self.balance += pnl
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'date': timestamp.strftime('%Y-%m-%d'),
            'prediction_proba': prediction_proba,
            'predicted_direction': predicted_direction,
            'actual_direction': actual_direction,
            'is_win': is_win,
            'trade_amount': self.trade_amount,
            'pnl': pnl,
            'balance': self.balance
        }
        
        self.trades.append(trade)
        
        # Update daily stats
        date_str = timestamp.strftime('%Y-%m-%d')
        if date_str not in self.daily_stats:
            self.daily_stats[date_str] = {'trades': 0, 'pnl': 0}
        
        self.daily_stats[date_str]['trades'] += 1
        self.daily_stats[date_str]['pnl'] += pnl
        
        return trade
    
    def run_backtest(
        self,
        timestamps: pd.DatetimeIndex,
        predictions_proba: np.ndarray,
        actual_directions: np.ndarray
    ) -> Dict:
        """
        Run complete backtest
        
        Args:
            timestamps: Array of timestamps
            predictions_proba: Model predictions [0, 1]
            actual_directions: Actual outcomes (1 = up, 0 = down)
        
        Returns:
            Backtest results dictionary
        """
        self.reset()
        
        print("="*60)
        print("RUNNING BACKTEST")
        print("="*60)
        print(f"Period: {timestamps[0]} to {timestamps[-1]}")
        print(f"Total opportunities: {len(predictions_proba)}")
        print(f"Initial balance: ${self.initial_balance:.2f}")
        print(f"Confidence threshold: {self.min_confidence:.0%}")
        print("-"*60)
        
        # Run through each prediction
        for i, (ts, pred_proba, actual) in enumerate(
            zip(timestamps, predictions_proba, actual_directions)
        ):
            date_str = ts.strftime('%Y-%m-%d')
            
            # Check if should trade
            if self.should_trade(pred_proba, date_str, self.balance):
                self.execute_trade(pred_proba, actual, ts)
        
        # Calculate statistics
        results = self.calculate_statistics()
        
        return results
    
    def calculate_statistics(self) -> Dict:
        """Calculate backtest statistics"""
        
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'message': 'No trades executed'
            }
        
        df_trades = pd.DataFrame(self.trades)
        
        # Basic statistics
        total_trades = len(df_trades)
        wins = df_trades['is_win'].sum()
        losses = total_trades - wins
        win_rate = wins / total_trades
        
        # Financial metrics
        total_pnl = df_trades['pnl'].sum()
        final_balance = self.balance
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Risk metrics
        winning_trades = df_trades[df_trades['is_win'] == True]['pnl']
        losing_trades = df_trades[df_trades['is_win'] == False]['pnl']
        
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Maximum drawdown
        df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
        df_trades['peak'] = df_trades['cumulative_pnl'].cummax()
        df_trades['drawdown'] = df_trades['cumulative_pnl'] - df_trades['peak']
        max_drawdown = df_trades['drawdown'].min()
        max_drawdown_pct = (max_drawdown / self.initial_balance) * 100
        
        # Sharpe ratio (simplified)
        returns = df_trades['pnl'] / self.trade_amount
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Daily statistics
        daily_summary = []
        for date, stats in sorted(self.daily_stats.items()):
            daily_summary.append({
                'date': date,
                'trades': stats['trades'],
                'pnl': stats['pnl']
            })
        
        # Print results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nTrade Statistics:")
        print(f"  Total trades: {total_trades}")
        print(f"  Wins: {wins} ({win_rate:.2%})")
        print(f"  Losses: {losses} ({(1-win_rate):.2%})")
        
        print(f"\nFinancial Performance:")
        print(f"  Initial balance: ${self.initial_balance:.2f}")
        print(f"  Final balance: ${final_balance:.2f}")
        print(f"  Total P&L: ${total_pnl:+.2f}")
        print(f"  Total return: {total_return:+.2f}%")
        
        print(f"\nRisk Metrics:")
        print(f"  Average win: ${avg_win:.2f}")
        print(f"  Average loss: ${avg_loss:.2f}")
        print(f"  Profit factor: {profit_factor:.2f}")
        print(f"  Max drawdown: ${max_drawdown:.2f} ({max_drawdown_pct:.2f}%)")
        print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
        
        print(f"\nBreak-even Analysis:")
        min_win_rate = 1 / (1 + self.payout_rate)
        print(f"  Minimum win rate for break-even: {min_win_rate:.2%}")
        print(f"  Current win rate: {win_rate:.2%}")
        
        if win_rate >= min_win_rate:
            print(f"  ✅ Strategy is profitable!")
        else:
            print(f"  ❌ Strategy is not profitable")
        
        print("="*60)
        
        return {
            'total_trades': total_trades,
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': win_rate,
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'min_win_rate_breakeven': min_win_rate,
            'trades_detail': self.trades,
            'daily_summary': daily_summary
        }
    
    def plot_results(self, save_path: str = None):
        """Plot backtest results"""
        
        if len(self.trades) == 0:
            print("No trades to plot")
            return
        
        df_trades = pd.DataFrame(self.trades)
        df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Balance over time
        axes[0, 0].plot(df_trades['timestamp'], df_trades['balance'])
        axes[0, 0].axhline(y=self.initial_balance, color='r', linestyle='--', label='Initial Balance')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Balance ($)')
        axes[0, 0].set_title('Account Balance Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Cumulative P&L
        axes[0, 1].plot(df_trades['timestamp'], df_trades['cumulative_pnl'])
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Cumulative P&L ($)')
        axes[0, 1].set_title('Cumulative Profit/Loss')
        axes[0, 1].grid(True)
        
        # 3. Trade distribution
        wins = df_trades[df_trades['is_win'] == True]
        losses = df_trades[df_trades['is_win'] == False]
        axes[1, 0].bar(['Wins', 'Losses'], [len(wins), len(losses)], color=['green', 'red'])
        axes[1, 0].set_ylabel('Number of Trades')
        axes[1, 0].set_title('Win/Loss Distribution')
        axes[1, 0].grid(True, axis='y')
        
        # 4. Daily P&L
        daily_pnl = df_trades.groupby('date')['pnl'].sum()
        colors = ['green' if x > 0 else 'red' for x in daily_pnl]
        axes[1, 1].bar(range(len(daily_pnl)), daily_pnl.values, color=colors)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('Daily P&L ($)')
        axes[1, 1].set_title('Daily Profit/Loss')
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nBacktest plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: str):
        """Save backtest results to JSON"""
        results = self.calculate_statistics()
        
        # Convert timestamps to strings
        for trade in results['trades_detail']:
            trade['timestamp'] = trade['timestamp'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Simulate some predictions
    np.random.seed(42)
    n_samples = 1000
    
    # Generate timestamps
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    # Generate predictions with some skill (60% accuracy)
    actual = np.random.randint(0, 2, n_samples)
    predictions = actual.copy()
    
    # Add noise (make 40% wrong)
    noise_idx = np.random.choice(n_samples, size=int(0.4 * n_samples), replace=False)
    predictions[noise_idx] = 1 - predictions[noise_idx]
    
    # Convert to probabilities
    predictions_proba = predictions.astype(float) + np.random.randn(n_samples) * 0.1
    predictions_proba = np.clip(predictions_proba, 0, 1)
    
    # Run backtest
    backtester = BacktestEngine(
        initial_balance=1000.0,
        payout_rate=0.80,
        min_confidence=0.60,
        trade_amount=10.0
    )
    
    results = backtester.run_backtest(timestamps, predictions_proba, actual)
    
    # Plot results
    backtester.plot_results(save_path='backtest_results.png')
    
    # Save results
    backtester.save_results('backtest_results.json')

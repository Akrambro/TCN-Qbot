"""
Risk Management Module

Implements risk management rules and position sizing strategies
"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime


class RiskManager:
    """
    Manages trading risk and position sizing
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        max_risk_per_trade: float = 0.02,  # 2% of balance
        max_daily_trades: int = 100,
        max_daily_loss_pct: float = 0.20,  # 20% of balance
        min_confidence: float = 0.60,
        kelly_fraction: float = 0.25  # Conservative Kelly
    ):
        """
        Initialize risk manager
        
        Args:
            initial_balance: Starting capital
            max_risk_per_trade: Maximum risk per trade as fraction of balance
            max_daily_trades: Maximum number of trades per day
            max_daily_loss_pct: Maximum daily loss as fraction of balance
            min_confidence: Minimum prediction confidence to trade
            kelly_fraction: Fraction of Kelly criterion to use (0.25 = quarter Kelly)
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss_pct = max_daily_loss_pct
        self.min_confidence = min_confidence
        self.kelly_fraction = kelly_fraction
        
        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        
        # Statistics
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
    
    def reset_daily_counters(self, current_date: str):
        """Reset daily counters at start of new day"""
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = current_date
    
    def update_balance(self, new_balance: float):
        """Update current balance"""
        self.current_balance = new_balance
    
    def update_statistics(self, is_win: bool, pnl: float):
        """Update trade statistics"""
        self.total_trades += 1
        self.daily_trades += 1
        self.daily_pnl += pnl
        
        if is_win:
            self.wins += 1
        else:
            self.losses += 1
    
    def get_win_rate(self) -> float:
        """Calculate current win rate"""
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades
    
    def calculate_kelly_position(
        self,
        win_rate: float,
        payout_rate: float
    ) -> float:
        """
        Calculate Kelly criterion position size
        
        Kelly formula for binary options:
        f = (p * (1 + b) - 1) / b
        
        where:
        - f = fraction of capital to bet
        - p = probability of winning
        - b = payout rate (e.g., 0.80)
        
        Args:
            win_rate: Historical win rate
            payout_rate: Payout ratio
        
        Returns:
            Fraction of balance to risk
        """
        if win_rate <= 0 or payout_rate <= 0:
            return 0.0
        
        # Kelly criterion
        kelly = (win_rate * (1 + payout_rate) - 1) / payout_rate
        
        # Apply conservative fraction (quarter Kelly)
        kelly = max(0.0, kelly * self.kelly_fraction)
        
        # Cap at max risk per trade
        kelly = min(kelly, self.max_risk_per_trade)
        
        return kelly
    
    def calculate_position_size(
        self,
        prediction_confidence: float,
        payout_rate: float,
        sizing_method: str = 'fixed'
    ) -> float:
        """
        Calculate position size based on selected method
        
        Args:
            prediction_confidence: Model confidence [0, 1]
            payout_rate: Payout ratio
            sizing_method: 'fixed', 'kelly', 'confidence', 'percentage'
        
        Returns:
            Position size in dollars
        """
        if sizing_method == 'fixed':
            # Fixed dollar amount
            return self.current_balance * self.max_risk_per_trade
        
        elif sizing_method == 'kelly':
            # Kelly criterion based on historical performance
            win_rate = self.get_win_rate()
            if win_rate > 0:
                kelly_fraction = self.calculate_kelly_position(win_rate, payout_rate)
                return self.current_balance * kelly_fraction
            else:
                return self.current_balance * self.max_risk_per_trade
        
        elif sizing_method == 'confidence':
            # Scale with prediction confidence
            confidence_score = abs(prediction_confidence - 0.5) * 2  # [0, 1]
            size_fraction = self.max_risk_per_trade * confidence_score
            return self.current_balance * size_fraction
        
        elif sizing_method == 'percentage':
            # Simple percentage of balance
            return self.current_balance * self.max_risk_per_trade
        
        else:
            raise ValueError(f"Unknown sizing method: {sizing_method}")
    
    def should_trade(
        self,
        prediction_confidence: float,
        current_date: str
    ) -> Dict[str, any]:
        """
        Determine if trade should be placed based on risk rules
        
        Args:
            prediction_confidence: Model prediction confidence
            current_date: Current date string
        
        Returns:
            Dictionary with decision and reason
        """
        # Reset daily counters if new day
        self.reset_daily_counters(current_date)
        
        # Check confidence threshold
        if not (prediction_confidence >= self.min_confidence or 
                prediction_confidence <= (1 - self.min_confidence)):
            return {
                'should_trade': False,
                'reason': f'Confidence {prediction_confidence:.2f} below threshold {self.min_confidence:.2f}',
                'confidence': prediction_confidence
            }
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return {
                'should_trade': False,
                'reason': f'Daily trade limit reached ({self.daily_trades}/{self.max_daily_trades})',
                'daily_trades': self.daily_trades
            }
        
        # Check daily loss limit
        max_daily_loss = self.initial_balance * self.max_daily_loss_pct
        if self.daily_pnl <= -max_daily_loss:
            return {
                'should_trade': False,
                'reason': f'Daily loss limit reached (${self.daily_pnl:.2f} / ${-max_daily_loss:.2f})',
                'daily_pnl': self.daily_pnl
            }
        
        # Check if balance sufficient
        min_position = self.current_balance * 0.01  # At least 1% of balance
        if self.current_balance < min_position:
            return {
                'should_trade': False,
                'reason': f'Insufficient balance (${self.current_balance:.2f})',
                'balance': self.current_balance
            }
        
        # All checks passed
        return {
            'should_trade': True,
            'reason': 'All risk checks passed',
            'confidence': prediction_confidence,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'balance': self.current_balance
        }
    
    def get_status(self) -> Dict:
        """Get current risk management status"""
        win_rate = self.get_win_rate()
        
        return {
            'balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_return_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'max_daily_trades': self.max_daily_trades,
            'max_daily_loss': self.initial_balance * self.max_daily_loss_pct
        }


class PortfolioManager:
    """
    Manages multiple trading strategies or assets
    """
    
    def __init__(
        self,
        total_capital: float = 10000.0,
        max_correlated_trades: int = 3
    ):
        """
        Initialize portfolio manager
        
        Args:
            total_capital: Total capital to allocate
            max_correlated_trades: Maximum number of correlated trades
        """
        self.total_capital = total_capital
        self.max_correlated_trades = max_correlated_trades
        self.allocations = {}
        self.active_trades = {}
    
    def allocate_capital(
        self,
        strategy_name: str,
        allocation_pct: float
    ):
        """
        Allocate capital to a strategy
        
        Args:
            strategy_name: Name of strategy
            allocation_pct: Percentage of capital to allocate (0-1)
        """
        if allocation_pct < 0 or allocation_pct > 1:
            raise ValueError("Allocation must be between 0 and 1")
        
        self.allocations[strategy_name] = {
            'allocation_pct': allocation_pct,
            'capital': self.total_capital * allocation_pct,
            'trades': 0,
            'pnl': 0.0
        }
    
    def get_available_capital(self, strategy_name: str) -> float:
        """Get available capital for strategy"""
        if strategy_name not in self.allocations:
            return 0.0
        
        return self.allocations[strategy_name]['capital']
    
    def check_correlation_limit(
        self,
        asset: str,
        correlated_assets: list
    ) -> bool:
        """
        Check if adding trade would exceed correlation limit
        
        Args:
            asset: Asset to trade
            correlated_assets: List of correlated assets
        
        Returns:
            True if trade allowed, False otherwise
        """
        # Count active trades in correlated assets
        correlated_active = sum(
            1 for a in correlated_assets 
            if a in self.active_trades
        )
        
        return correlated_active < self.max_correlated_trades


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("RISK MANAGEMENT DEMO")
    print("="*60)
    
    # Initialize risk manager
    rm = RiskManager(
        initial_balance=1000.0,
        max_risk_per_trade=0.02,
        max_daily_trades=50,
        max_daily_loss_pct=0.20,
        min_confidence=0.60
    )
    
    print(f"\nInitial configuration:")
    print(f"  Balance: ${rm.current_balance:.2f}")
    print(f"  Max risk per trade: {rm.max_risk_per_trade:.1%}")
    print(f"  Max daily trades: {rm.max_daily_trades}")
    print(f"  Max daily loss: {rm.max_daily_loss_pct:.1%}")
    print(f"  Min confidence: {rm.min_confidence:.1%}")
    
    # Simulate some trades
    print("\n" + "-"*60)
    print("Simulating trades...")
    print("-"*60)
    
    # Trade 1: High confidence
    decision = rm.should_trade(0.75, "2024-01-01")
    print(f"\nTrade 1 (confidence: 0.75):")
    print(f"  Decision: {'✅ TRADE' if decision['should_trade'] else '❌ SKIP'}")
    print(f"  Reason: {decision['reason']}")
    
    if decision['should_trade']:
        position_size = rm.calculate_position_size(0.75, 0.80, 'fixed')
        print(f"  Position size: ${position_size:.2f}")
        
        # Simulate win
        rm.update_statistics(True, position_size * 0.80)
        rm.update_balance(rm.current_balance + position_size * 0.80)
    
    # Trade 2: Low confidence
    decision = rm.should_trade(0.55, "2024-01-01")
    print(f"\nTrade 2 (confidence: 0.55):")
    print(f"  Decision: {'✅ TRADE' if decision['should_trade'] else '❌ SKIP'}")
    print(f"  Reason: {decision['reason']}")
    
    # Get status
    print("\n" + "-"*60)
    print("Current Status:")
    print("-"*60)
    status = rm.get_status()
    for key, value in status.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

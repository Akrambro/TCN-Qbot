"""
Data Preprocessing and Feature Engineering for Forex Trading

Based on AboutBot.pdf methodology:
- Resampling and alignment of timestamps
- Normalization/Scaling (returns, z-score)
- Multivariate channels (OHLC + volume)
- Lagged sequences for sliding windows
- Technical indicators as additional features
- Time-based features for seasonality
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ForexDataPreprocessor:
    """
    Preprocessing pipeline for Forex data
    """
    
    def __init__(
        self,
        use_log_returns: bool = True,
        scaler_type: str = 'standard',  # 'standard' or 'minmax'
        add_technical_indicators: bool = True
    ):
        self.use_log_returns = use_log_returns
        self.scaler_type = scaler_type
        self.add_technical_indicators = add_technical_indicators
        self.scaler = None
        self.feature_columns = None
    
    def load_and_prepare(
        self, 
        filepath: str,
        timestamp_col: str = 'timestamp',
        resample_freq: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load and prepare raw data
        
        Args:
            filepath: Path to CSV file
            timestamp_col: Name of timestamp column
            resample_freq: Resample frequency (e.g., '1min', '5min')
        
        Returns:
            Prepared DataFrame
        """
        # Load data
        df = pd.read_csv(filepath)
        
        # Convert timestamp
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)
        
        # Resample if needed
        if resample_freq:
            df = self._resample_ohlcv(df, resample_freq)
        
        # Forward fill missing values
        df = df.fillna(method='ffill').dropna()
        
        print(f"Loaded {len(df)} candles")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def _resample_ohlcv(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample OHLCV data to specified frequency"""
        resampled = pd.DataFrame()
        
        if 'open' in df.columns:
            resampled['open'] = df['open'].resample(freq).first()
        if 'high' in df.columns:
            resampled['high'] = df['high'].resample(freq).max()
        if 'low' in df.columns:
            resampled['low'] = df['low'].resample(freq).min()
        if 'close' in df.columns:
            resampled['close'] = df['close'].resample(freq).last()
        if 'volume' in df.columns:
            resampled['volume'] = df['volume'].resample(freq).sum()
        
        return resampled
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return features
        
        Log returns: r_t = ln(P_t / P_{t-1})
        Simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}
        """
        df = df.copy()
        
        if self.use_log_returns:
            # Log returns (more stable for modeling)
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
        else:
            # Simple percentage returns
            df['returns'] = df['close'].pct_change()
        
        # Returns for other prices
        if 'open' in df.columns:
            df['open_returns'] = df['open'].pct_change()
        if 'high' in df.columns:
            df['high_returns'] = df['high'].pct_change()
        if 'low' in df.columns:
            df['low_returns'] = df['low'].pct_change()
        
        return df
    
    def add_technical_indicators_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators as described in the research
        
        Features include:
        - Moving averages (SMA 5, 15)
        - RSI
        - MACD
        - Volatility (rolling std)
        """
        df = df.copy()
        
        # Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_15'] = df['close'].rolling(window=15).mean()
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Volatility (ATR approximation)
        df['volatility'] = df['close'].rolling(window=14).std()
        
        # Price momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        return df
    
    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced price action features for better signal detection
        
        These features capture candlestick patterns, volatility regimes,
        and market microstructure that technical indicators miss.
        
        Features include:
        - Candle body/wick ratios
        - Consecutive streak counters
        - Relative position within day/session
        - Volatility regime indicators
        - Higher timeframe context
        """
        df = df.copy()
        
        # === CANDLE BODY & WICK FEATURES ===
        # Body size relative to full range
        candle_range = df['high'] - df['low']
        candle_body = abs(df['close'] - df['open'])
        df['body_ratio'] = candle_body / (candle_range + 1e-8)  # Avoid division by zero
        
        # Upper wick (above body)
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        df['upper_wick_ratio'] = upper_shadow / (candle_range + 1e-8)
        
        # Lower wick (below body)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        df['lower_wick_ratio'] = lower_shadow / (candle_range + 1e-8)
        
        # Candle direction
        df['candle_direction'] = (df['close'] > df['open']).astype(int)  # 1=bullish, 0=bearish
        
        # Body size relative to recent average
        avg_body = candle_body.rolling(window=20).mean()
        df['body_size_ratio'] = candle_body / (avg_body + 1e-8)
        
        # === CONSECUTIVE STREAKS ===
        # Count consecutive green/red candles
        direction_change = df['candle_direction'].diff().ne(0)
        streak_id = direction_change.cumsum()
        df['candle_streak'] = df.groupby(streak_id).cumcount() + 1
        # Make streak negative for bearish streaks
        df['candle_streak'] = df['candle_streak'] * (df['candle_direction'] * 2 - 1)
        
        # === VOLATILITY REGIME FEATURES ===
        # ATR (Average True Range) - proper calculation
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # ATR ratio (current vs average) - detects volatility spikes
        avg_atr = df['atr'].rolling(window=50).mean()
        df['atr_ratio'] = df['atr'] / (avg_atr + 1e-8)
        
        # Volatility spike detector
        df['volatility_spike'] = (df['atr_ratio'] > 1.5).astype(int)
        
        # === PRICE POSITION FEATURES ===
        # Where is close relative to high-low range? (0=at low, 1=at high)
        df['close_position'] = (df['close'] - df['low']) / (candle_range + 1e-8)
        
        # Distance from recent high/low
        rolling_high = df['high'].rolling(window=20).max()
        rolling_low = df['low'].rolling(window=20).min()
        rolling_range = rolling_high - rolling_low
        df['dist_from_high'] = (rolling_high - df['close']) / (rolling_range + 1e-8)
        df['dist_from_low'] = (df['close'] - rolling_low) / (rolling_range + 1e-8)
        
        # === MOMENTUM & ACCELERATION ===
        # Price velocity (rate of change)
        df['price_velocity'] = df['close'].diff() / df['close'].shift(1)
        df['price_acceleration'] = df['price_velocity'].diff()
        
        # Volume acceleration (if volume available)
        if 'volume' in df.columns:
            df['volume_velocity'] = df['volume'].diff() / (df['volume'].shift(1) + 1e-8)
            # Volume surge detector
            avg_volume = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / (avg_volume + 1e-8)
            df['volume_surge'] = (df['volume_ratio'] > 2.0).astype(int)
        
        # === HIGHER TIMEFRAME CONTEXT ===
        # Longer-period trend indicators (simulating 5-min, 15-min views)
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        
        # Price relative to higher timeframe MAs
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 1e-8)
        df['price_vs_sma100'] = (df['close'] - df['sma_100']) / (df['sma_100'] + 1e-8)
        
        # Higher timeframe trend
        df['sma50_slope'] = df['sma_50'].diff(5)  # 5-bar slope
        df['sma100_slope'] = df['sma_100'].diff(10)  # 10-bar slope
        
        # === RECENT PRICE ACTION ===
        # Last 3 candles' characteristics (helps capture short-term patterns)
        for i in [1, 2, 3]:
            df[f'body_ratio_lag{i}'] = df['body_ratio'].shift(i)
            df[f'close_position_lag{i}'] = df['close_position'].shift(i)
        
        return df
    
    def add_lagged_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5]
    ) -> pd.DataFrame:
        """
        Add lagged features as described in AboutBot.pdf
        
        Args:
            df: DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for intraday seasonality
        
        Features:
        - Hour of day (cyclical encoding)
        - Day of week
        - Minute of day
        """
        df = df.copy()
        
        if isinstance(df.index, pd.DatetimeIndex):
            # Hour (cyclical encoding)
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            
            # Day of week
            df['day_of_week'] = df.index.dayofweek
            
            # Minute of day
            df['minute_of_day'] = df.index.hour * 60 + df.index.minute
            df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)
            
            # Market session indicators (example for European/US sessions)
            # London: 08:00-16:00 UTC
            df['london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
            # New York: 13:00-21:00 UTC
            df['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
        
        return df
    
    def create_target(
        self, 
        df: pd.DataFrame,
        prediction_horizon: int = 1
    ) -> pd.DataFrame:
        """
        Create binary target for classification
        
        Target = 1 if next close > current close (CALL wins)
        Target = 0 if next close <= current close (PUT wins)
        
        Args:
            df: DataFrame with 'close' column
            prediction_horizon: How many periods ahead to predict
        """
        df = df.copy()
        
        # Shift close price by prediction horizon
        df['next_close'] = df['close'].shift(-prediction_horizon)
        
        # Binary target: 1 = up, 0 = down
        df['target'] = (df['next_close'] > df['close']).astype(int)
        
        # Remove last rows where target is NaN
        df = df.dropna(subset=['target'])
        
        return df
    
    def normalize_features(
        self, 
        df: pd.DataFrame,
        feature_columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using StandardScaler or MinMaxScaler
        
        Args:
            df: DataFrame
            feature_columns: Columns to normalize
            fit: Whether to fit the scaler (True for training, False for inference)
        """
        df = df.copy()
        
        # Initialize scaler if needed
        if fit or self.scaler is None:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            
            # Fit and transform
            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
            self.feature_columns = feature_columns
        else:
            # Transform only
            df[feature_columns] = self.scaler.transform(df[feature_columns])
        
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        sequence_length: int = 60,
        target_column: str = 'target'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for TCN input
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            sequence_length: Length of each sequence (window size)
            target_column: Name of target column
        
        Returns:
            X: Array of shape (n_samples, n_features, sequence_length)
            y: Array of shape (n_samples,)
        """
        # Extract feature matrix
        features = df[feature_columns].values
        targets = df[target_column].values
        
        X_sequences = []
        y_sequences = []
        
        # Create sliding windows
        for i in range(len(df) - sequence_length):
            # Extract sequence
            sequence = features[i:i + sequence_length]
            target = targets[i + sequence_length]
            
            X_sequences.append(sequence)
            y_sequences.append(target)
        
        # Convert to numpy arrays
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        # Transpose to (n_samples, n_features, sequence_length)
        X = np.transpose(X, (0, 2, 1))
        
        return X, y
    
    def full_pipeline(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Raw DataFrame with OHLCV data
            sequence_length: Window size for sequences
            prediction_horizon: How many steps ahead to predict
            fit: Whether to fit scalers (True for training)
        
        Returns:
            X: Input sequences
            y: Target labels
            feature_columns: List of feature names
        """
        # Add returns
        df = self.add_returns(df)
        
        # Add technical indicators
        if self.add_technical_indicators:
            df = self.add_technical_indicators_basic(df)
            # Add advanced price action features
            df = self.add_price_action_features(df)
        
        # Add lagged features
        df = self.add_lagged_features(df, ['close', 'returns'], lags=[1, 2, 3, 5])
        
        # Add time features
        df = self.add_time_features(df)
        
        # Create target
        df = self.create_target(df, prediction_horizon)
        
        # Drop NaN values
        df = df.dropna()
        
        # Select feature columns (exclude OHLCV, target, and timestamp)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 
                       'next_close', 'minute_of_day', 'timestamp', 'date', 'time', 'datetime']
        feature_columns = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Normalize features
        df = self.normalize_features(df, feature_columns, fit=fit)
        
        # Create sequences
        X, y = self.create_sequences(df, feature_columns, sequence_length)
        
        print(f"\nPreprocessing complete:")
        print(f"Total samples: {len(X)}")
        print(f"Input shape: {X.shape}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Target distribution: {y.mean():.2%} up, {1-y.mean():.2%} down")
        
        return X, y, feature_columns


# Example usage
if __name__ == "__main__":
    # Example: Load and preprocess data
    preprocessor = ForexDataPreprocessor(
        use_log_returns=True,
        scaler_type='standard',
        add_technical_indicators=True
    )
    
    # Simulate loading data (replace with actual file)
    # df = preprocessor.load_and_prepare('data/EURUSD_1min.csv')
    
    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5000, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 1.1000 + np.cumsum(np.random.randn(5000) * 0.0001),
        'high': 1.1000 + np.cumsum(np.random.randn(5000) * 0.0001) + 0.0005,
        'low': 1.1000 + np.cumsum(np.random.randn(5000) * 0.0001) - 0.0005,
        'close': 1.1000 + np.cumsum(np.random.randn(5000) * 0.0001),
        'volume': np.random.randint(100, 1000, 5000)
    })
    df = df.set_index('timestamp')
    
    # Run full pipeline
    X, y, feature_columns = preprocessor.full_pipeline(
        df,
        sequence_length=60,
        prediction_horizon=1,
        fit=True
    )
    
    print(f"\nFeature columns ({len(feature_columns)}):")
    for i, col in enumerate(feature_columns[:10]):
        print(f"  {i+1}. {col}")
    if len(feature_columns) > 10:
        print(f"  ... and {len(feature_columns) - 10} more")

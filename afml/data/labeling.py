import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Union, List, Dict
from scipy.stats import norm
from joblib import Parallel, delayed

class DailyVolatility:
    """
    Compute daily volatility to scale price targets for triple-barrier method.
    """
    
    @staticmethod
    def get_daily_vol(prices: pd.Series, span: int = 100) -> pd.Series:
        """
        Compute daily volatility using exponentially weighted moving average.
        """
        returns = prices.pct_change().dropna()
        vol = returns.ewm(span=span).std() * np.sqrt(252)
        return vol


class TripleBarrierLabeling:
    """
    Implementation of the Triple Barrier Method for labeling financial data.
    """
    
    def __init__(self, prices: pd.Series, events: pd.DataFrame, 
                 pt_sl: Tuple[float, float], molecule: List[pd.Timestamp] = None,
                 min_ret: float = 0.0, num_threads: int = 1, t_events: bool = True,
                 side_prediction: pd.Series = None):
        self.prices = prices
        self.events = events.copy()
        if 'tl' not in self.events.columns:
            self.events['tl'] = self.events.index
        if 'trgt' not in self.events.columns:
            raise ValueError("'trgt' column is required in events DataFrame")
        self.pt_sl = pt_sl
        self.molecule = molecule if molecule is not None else self.events.index
        self.min_ret = min_ret
        self.num_threads = num_threads
        self.t_events = t_events
        self.side_prediction = side_prediction
        
    def apply_triple_barrier(self, timestamp: pd.Timestamp) -> pd.Series:
        try:
            event = self.events.loc[timestamp]
            tl = event.tl
            trgt = event.trgt
            
            if 'side' in self.events.columns:
                side = event.side
            elif self.side_prediction is not None:
                side = self.side_prediction.loc[timestamp]
            else:
                side = 1
            
            if trgt < self.min_ret:
                return pd.Series([0, tl, 0], index=['ret', 't_touch', 'label'])
                
            upper_barrier = self.pt_sl[0] * trgt
            lower_barrier = -self.pt_sl[1] * trgt
            
            if isinstance(tl, pd.Timestamp) and self.t_events:
                price_window = self.prices[timestamp:tl]
            else:
                price_window = self.prices[timestamp:]
                
            if price_window.empty:
                return pd.Series([0, self.prices.index[-1], 0], index=['ret', 't_touch', 'label'])
            
            price_t0 = self.prices.loc[timestamp]
            price_returns = (price_window / price_t0 - 1) * side
            
            hit_upper = price_returns >= upper_barrier
            hit_lower = price_returns <= lower_barrier
            
            if hit_upper.any():
                t_upper = price_returns[hit_upper].index[0]
                if hit_lower.any():
                    t_lower = price_returns[hit_lower].index[0]
                    if t_upper < t_lower:
                        return pd.Series([upper_barrier, t_upper, 1], index=['ret', 't_touch', 'label'])
                    else:
                        return pd.Series([lower_barrier, t_lower, -1], index=['ret', 't_touch', 'label'])
                else:
                    return pd.Series([upper_barrier, t_upper, 1], index=['ret', 't_touch', 'label'])
            
            elif hit_lower.any():
                t_lower = price_returns[hit_lower].index[0]
                return pd.Series([lower_barrier, t_lower, -1], index=['ret', 't_touch', 'label'])
            
            else:
                if price_returns.empty:
                    return pd.Series([0, self.prices.index[-1], 0], index=['ret', 't_touch', 'label'])
                return pd.Series([price_returns.iloc[-1], price_returns.index[-1], 0], index=['ret', 't_touch', 'label'])
            
        except Exception as e:
            return pd.Series([0, timestamp, 0], index=['ret', 't_touch', 'label'])
        
    def get_events(self) -> pd.DataFrame:
        if self.num_threads > 1:
            events_results = Parallel(n_jobs=self.num_threads)(
                delayed(self.apply_triple_barrier)(t_idx) for t_idx in self.molecule
            )
            events = []
            for i, t_idx in enumerate(self.molecule):
                result = events_results[i].copy()
                result['tl'] = t_idx
                events.append(result)
        else:
            events = []
            for t_idx in self.molecule:
                result = self.apply_triple_barrier(t_idx)
                result['tl'] = t_idx
                events.append(result)
            
        if not events:
            return pd.DataFrame(columns=['ret', 't_touch', 'label', 'tl'])
            
        events_df = pd.DataFrame(events)
        events_df.set_index('tl', inplace=True)
        return events_df
    
    @staticmethod
    def get_bins(triple_barrier_events: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        events = triple_barrier_events.copy()
        events['bin'] = np.sign(events['ret'])
        
        if 'side' in events.columns:
            events.loc[events['side'] * events['ret'] > 0, 'bin'] = 1
            events.loc[events['side'] * events['ret'] <= 0, 'bin'] = 0
            
        return events


class MetaLabeling:
    """
    Functions for meta-labeling and bet sizing.
    """

    @staticmethod
    def calculate_prob_metrics(probabilities: np.ndarray, labels: np.ndarray, 
                               pred_thresh: float = 0.5) -> Dict[str, float]:
        pred_labels = (probabilities >= pred_thresh).astype(int)
        
        accuracy = np.mean(pred_labels == labels)
        
        true_pos = np.sum((pred_labels == 1) & (labels == 1))
        false_pos = np.sum((pred_labels == 1) & (labels == 0))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'true_positives': true_pos,
            'false_positives': false_pos
        }

    @staticmethod
    def bet_size(prob: float, kelly_fraction: float = 0.5) -> float:
        if prob == 0.5:
            return 0
        
        if prob > 0.5:
            bet = (prob - 0.5) / (prob * (1 - prob))
        else:
            bet = -(0.5 - prob) / (prob * (1 - prob))
            
        return kelly_fraction * bet

    @staticmethod
    def plot_precision_vs_accuracy(probabilities: np.ndarray, labels: np.ndarray) -> None:
        thresholds = np.linspace(0.5, 1.0, 26)
        metrics = [MetaLabeling.calculate_prob_metrics(probabilities, labels, t) for t in thresholds]
        
        accuracy = [m['accuracy'] for m in metrics]
        precision = [m['precision'] for m in metrics]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(thresholds, accuracy, 'b-', label='Accuracy')
        ax1.set_xlabel('Prediction Threshold')
        ax1.set_ylabel('Accuracy', color='b')
        ax1.tick_params('y', colors='b')
        
        ax2 = ax1.twinx()
        ax2.plot(thresholds, precision, 'r-', label='Precision')
        ax2.set_ylabel('Precision', color='r')
        ax2.tick_params('y', colors='r')
        
        plt.title('Precision and Accuracy vs. Prediction Threshold')
        fig.tight_layout()
        plt.show() 


class FixedHorizonLabeling:
    """
    Labels events based on fixed percentage changes with a time limit (vertical barrier).
    This is a simplified application of the triple-barrier method.

    - Label 1: if the price increases by upper_pct.
    - Label -1: if the price decreases by lower_pct.
    - Label 0: if neither of the above happens within the time_horizon.
    """
    
    def __init__(self, prices: pd.Series, upper_pct: float, lower_pct: float, 
                 time_horizon: pd.Timedelta, num_threads: int = 1):
        self.prices = prices
        self.upper_pct = upper_pct
        self.lower_pct = lower_pct
        self.time_horizon = time_horizon
        self.num_threads = num_threads

    def _apply_label(self, timestamp: pd.Timestamp) -> pd.Series:
        price_t0 = self.prices.loc[timestamp]
        
        vertical_barrier = timestamp + self.time_horizon
        if vertical_barrier > self.prices.index[-1]:
            vertical_barrier = self.prices.index[-1]

        price_window = self.prices.loc[timestamp:vertical_barrier]
        
        if len(price_window) < 2:
            return pd.Series([0, timestamp, 0], index=['ret', 't_touch', 'label'])
            
        returns = (price_window / price_t0) - 1
        
        hit_upper = returns[returns >= self.upper_pct]
        hit_lower = returns[returns <= -self.lower_pct]
        
        t_upper = hit_upper.index[0] if not hit_upper.empty else None
        t_lower = hit_lower.index[0] if not hit_lower.empty else None
        
        if t_upper and t_lower:
            if t_upper <= t_lower:
                return pd.Series([returns.loc[t_upper], t_upper, 1], index=['ret', 't_touch', 'label'])
            else:
                return pd.Series([returns.loc[t_lower], t_lower, -1], index=['ret', 't_touch', 'label'])
        elif t_upper:
            return pd.Series([returns.loc[t_upper], t_upper, 1], index=['ret', 't_touch', 'label'])
        elif t_lower:
            return pd.Series([returns.loc[t_lower], t_lower, -1], index=['ret', 't_touch', 'label'])
        else:
            last_ret = returns.iloc[-1]
            return pd.Series([last_ret, price_window.index[-1], 0], index=['ret', 't_touch', 'label'])

    def get_labels(self, timestamps: List[pd.Timestamp]) -> pd.DataFrame:
        if self.num_threads > 1:
            results = Parallel(n_jobs=self.num_threads)(
                delayed(self._apply_label)(t) for t in timestamps
            )
        else:
            results = [self._apply_label(t) for t in timestamps]
            
        results_df = pd.DataFrame(results, index=timestamps)
        return results_df


class PeriodReturnLabeling:
    """
    Assigns labels based on price changes over a fixed period, if a certain
    percentage change is met.
    - Label 1: if price change > pct_change.
    - Label -1: if price change < -pct_change.
    - Label 0: otherwise.
    """
    @staticmethod
    def get_labels(prices: pd.Series, time_horizon: int, pct_change: float = 0.0) -> pd.Series:
        forward_returns = prices.pct_change(periods=time_horizon).shift(-time_horizon)
        
        labels = pd.Series(0, index=forward_returns.index)
        labels.loc[forward_returns > pct_change] = 1
        labels.loc[forward_returns < -pct_change] = -1
        
        return labels.dropna()

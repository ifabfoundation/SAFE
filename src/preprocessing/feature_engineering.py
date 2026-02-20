"""
Feature Engineering per Bonfiglioli Anomaly Detection Pipeline

Modulo semplificato per la creazione di features derivate.
Applica rolling statistics solo a temperature e vibrazioni.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering per segnali temporali.
    
    Crea features rolling per temperature e vibrazioni:
    - Rolling mean
    - Rolling std
    - Rolling min
    - Rolling max
    
    I segnali fieldbus (fb_*) non vengono processati.
    """
    
    def __init__(self, config: Union[str, Path, Dict]):
        """
        Inizializza il feature engineer.
        
        Args:
            config: Path al file YAML o dizionario configurazione
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config_path = Path(config)
            self.config = self._load_config()
        
        # Parametri rolling
        fe_config = self.config.get('feature_engineering', {})
        self.window_size = fe_config.get('window_size', 50)
        self.min_periods = fe_config.get('min_periods', 1)
        
        logger.info(f"FeatureEngineer inizializzato (window={self.window_size})")
    
    def _load_config(self) -> Dict:
        """Carica configurazione da file YAML."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_signals_to_process(self, df: pd.DataFrame) -> List[str]:
        """
        Identifica i segnali da processare (temperature e vibrazioni).
        Esclude fieldbus (fb_*) e colonne temporali.
        
        Args:
            df: DataFrame con segnali
            
        Returns:
            Lista di nomi colonne da processare
        """
        exclude_prefixes = ['fb_']  # Fieldbus
        temporal_cols = ['datetime', 'time', 'time_relative']
        
        signals = []
        for col in df.columns:
            # Salta colonne temporali
            if col in temporal_cols:
                continue
            
            # Salta fieldbus
            if any(col.startswith(prefix) for prefix in exclude_prefixes):
                continue
            
            # Salta colonne non numeriche
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            signals.append(col)
        
        return signals
    
    def compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola tutte le features rolling per temperature e vibrazioni.
        
        Per ogni segnale calcola:
        - {signal}_mean: media mobile
        - {signal}_std: deviazione standard mobile
        - {signal}_min: minimo mobile
        - {signal}_max: massimo mobile
        
        Args:
            df: DataFrame con segnali
            
        Returns:
            DataFrame con features aggiunte
        """
        df_out = df.copy()
        signals = self._get_signals_to_process(df)
        
        logger.info(f"Calcolo rolling features per {len(signals)} segnali...")
        
        window = self.window_size
        min_periods = self.min_periods
        
        for signal in signals:
            # Rolling mean
            df_out[f"{signal}_mean"] = df_out[signal].rolling(
                window=window, min_periods=min_periods
            ).mean()
            
            # Rolling std
            df_out[f"{signal}_std"] = df_out[signal].rolling(
                window=window, min_periods=min_periods
            ).std()
            
            # Rolling min
            df_out[f"{signal}_min"] = df_out[signal].rolling(
                window=window, min_periods=min_periods
            ).min()
            
            # Rolling max
            df_out[f"{signal}_max"] = df_out[signal].rolling(
                window=window, min_periods=min_periods
            ).max()
            
            logger.debug(f"  {signal}: mean, std, min, max")
        
        return df_out
    
    def compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola features temporali dalla colonna datetime.
        
        Features create:
        - hour: ora del giorno (0-23)
        - day_of_week: giorno della settimana (0=lunedì, 6=domenica)
        
        Args:
            df: DataFrame con colonna 'datetime'
            
        Returns:
            DataFrame con features temporali aggiunte
        """
        df_out = df.copy()
        
        if 'datetime' not in df_out.columns:
            logger.warning("Colonna 'datetime' non trovata, skip features temporali")
            return df_out
        
        # Assicurati che datetime sia di tipo datetime
        if not pd.api.types.is_datetime64_any_dtype(df_out['datetime']):
            df_out['datetime'] = pd.to_datetime(df_out['datetime'])
        
        # Ora del giorno (0-23)
        df_out['hour'] = df_out['datetime'].dt.hour
        
        # Giorno della settimana (0=lunedì, 6=domenica)
        df_out['day_of_week'] = df_out['datetime'].dt.dayofweek
        
        logger.info("  Features temporali: hour, day_of_week")
        
        return df_out
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applica feature engineering completo.
        
        Alias per transform() per compatibilità con main.py.
        
        Args:
            df: DataFrame con segnali grezzi
            
        Returns:
            DataFrame con features aggiunte
        """
        return self.transform(df)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applica tutte le trasformazioni di feature engineering.
        
        Args:
            df: DataFrame con segnali grezzi
            
        Returns:
            DataFrame con features aggiunte
        """
        logger.info("Inizio feature engineering...")
        
        n_cols_start = len(df.columns)
        
        # Calcola features temporali (hour, day_of_week)
        df_out = self.compute_temporal_features(df)
        
        # Calcola features rolling
        df_out = self.compute_rolling_features(df_out)
        
        n_cols_end = len(df_out.columns)
        n_features_added = n_cols_end - n_cols_start
        
        logger.info(f"Feature engineering completato: {n_features_added} nuove features")
        logger.info(f"   Shape: {df.shape} -> {df_out.shape}")
        
        return df_out
    
    def get_feature_names(
        self, 
        df: pd.DataFrame,
        exclude_temporal: bool = True,
        only_engineered: bool = False
    ) -> List[str]:
        """
        Restituisce lista di nomi features.
        
        Args:
            df: DataFrame con features
            exclude_temporal: Se True, esclude datetime/time/time_relative
            only_engineered: Se True, restituisce solo features calcolate
            
        Returns:
            Lista di nomi colonne
        """
        temporal_cols = ['datetime', 'time', 'time_relative']
        engineered_suffixes = ['_mean', '_std', '_min', '_max']
        
        features = []
        for col in df.columns:
            if exclude_temporal and col in temporal_cols:
                continue
            
            if only_engineered:
                if any(col.endswith(suffix) for suffix in engineered_suffixes):
                    features.append(col)
            else:
                features.append(col)
        
        return features
    
    def get_feature_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restituisce statistiche sulle features.
        
        Args:
            df: DataFrame con features
            
        Returns:
            DataFrame con statistiche per feature
        """
        feature_cols = self.get_feature_names(df)
        
        info_list = []
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            info_list.append({
                'feature': col,
                'non_null': int(df[col].notna().sum()),
                'null': int(df[col].isna().sum()),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            })
        
        return pd.DataFrame(info_list)


def load_config(config_path: Union[str, Path]) -> Dict:
    """Utility per caricare configurazione YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 3:
        print("Usage: python feature_engineering.py <config.yaml> <input.csv>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    input_csv = sys.argv[2]
    
    # Carica dati
    print(f"Caricamento: {input_csv}")
    df = pd.read_csv(input_csv)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"   Shape: {df.shape}")
    
    # Feature engineering
    fe = FeatureEngineer(config_path)
    df_features = fe.transform(df)
    
    # Info
    print(f"\nFeatures create:")
    for col in fe.get_feature_names(df_features, only_engineered=True)[:10]:
        print(f"   - {col}")
    
    # Salva
    output_path = input_csv.replace('.csv', '_features.csv')
    df_features.to_csv(output_path, index=False)
    print(f"\nSalvato: {output_path}")

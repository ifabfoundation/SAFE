"""
Data Loader per Bonfiglioli Anomaly Detection Pipeline

Modulo per il caricamento e preprocessing dei dati da file H5 basato su configurazione YAML.
"""

import h5py
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BonfiglioliDataLoader:
    """
    Data loader per file H5 di Bonfiglioli con caricamento lazy e configurazione YAML.
    
    Features:
    - Caricamento lazy dei dati da file H5 (ottimizzato per performance)
    - Configurazione tramite file YAML
    - Estrazione di segnali multipli (vibrazione, temperatura, fieldbus)
    - Allineamento temporale tramite interpolazione
    - Progress bar per monitoraggio
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Inizializza il data loader con configurazione YAML.
        
        Args:
            config_path: Path al file di configurazione YAML
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.signals_config = self._build_signals_config()
        
        # Setup logging level
        log_level = getattr(logging, self.config['logging']['level'], logging.INFO)
        logger.setLevel(log_level)
        
        logger.info(f"DataLoader inizializzato con {len(self.signals_config)} segnali configurati")
    
    def _load_config(self) -> Dict:
        """Carica configurazione da file YAML."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configurazione caricata da {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Errore nel caricamento della configurazione: {e}")
            raise
    
    def _build_signals_config(self) -> Dict:
        """
        Costruisce dizionario di configurazione segnali dal formato YAML.
        
        Returns:
            Dict con formato {signal_name: (group, sensor, subgroup, column)}
        """
        signals_dict = {}
        
        # Processa ogni categoria di segnali
        for category in ['vibration', 'temperature', 'generic', 'fieldbus']:
            if category in self.config['signals']:
                for signal in self.config['signals'][category]:
                    signals_dict[signal['name']] = (
                        signal['group'],
                        signal['sensor'],
                        signal['subgroup'],
                        signal['column']
                    )
        
        logger.debug(f"Segnali configurati: {list(signals_dict.keys())}")
        return signals_dict
    
    def extract_time_series_lazy(
        self, 
        h5_file: h5py.File, 
        group_name: str, 
        signal_name: str,
        sub_group: str, 
        col_name: Optional[str] = None,
        return_datetime: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Estrae una time series direttamente dal file H5 senza conversione completa.
        
        Gestisce sia dataset diretti che gruppi con timestamp annidati.
        
        Args:
            h5_file: File H5 aperto (h5py.File)
            group_name: Nome del gruppo principale (es. 'Vibration')
            signal_name: Nome del segnale (es. 'AccRidMarcia_RMS')
            sub_group: Sottogruppo ('PreProcess' o 'RAW')
            col_name: Nome della colonna (None per RAW, 'None' per PreProcess/None)
            return_datetime: Se True, converte time in datetime, altrimenti mantiene secondi
            
        Returns:
            DataFrame con colonne ['time'/'datetime', 'value'] o None se errore
        """
        try:
            # Costruisci il path base H5
            if col_name:
                base_path = f"{group_name}/{signal_name}/{sub_group}/{col_name}"
            else:
                base_path = f"{group_name}/{signal_name}/{sub_group}"
            
            # Accedi al gruppo
            group_obj = h5_file[base_path]
            
            # Raccogli dati da tutti i timestamp
            all_times = []
            all_values = []
            
            # Se è un dataset diretto, estrailo
            if isinstance(group_obj, h5py.Dataset):
                dataset = group_obj[:]
                if dataset.ndim == 2 and dataset.shape[1] == 2:
                    all_times.extend(dataset[:, 0].tolist())
                    all_values.extend(dataset[:, 1].tolist())
                else:
                    logger.warning(f"Formato dataset non supportato per {signal_name}")
                    return None
            
            # Altrimenti è un gruppo con timestamp come chiavi
            elif isinstance(group_obj, h5py.Group):
                # IMPORTANTE: Gli offset nella colonna 0 sono ASSOLUTI dall'inizio acquisizione!
                # Le chiavi timestamp sono solo metadati organizzativi, NON base temporale
                # Quindi: NON sommare base_timestamp + offset, usa solo gli offset!
                
                # Trova la prima chiave timestamp per determinare la data di inizio acquisizione
                first_timestamp_key = None
                first_datetime = None
                
                for timestamp_key in group_obj.keys():
                    try:
                        # Parse: DD_MM_YYYY__HH_MM_SS
                        parts = timestamp_key.split('__')
                        if len(parts) == 2:
                            date_str = parts[0]  # DD_MM_YYYY
                            time_str = parts[1]  # HH_MM_SS
                            dt = pd.to_datetime(f"{date_str} {time_str}", format="%d_%m_%Y %H_%M_%S")
                            if first_datetime is None or dt < first_datetime:
                                first_datetime = dt
                                first_timestamp_key = timestamp_key
                    except:
                        pass
                
                # Itera su tutte le chiavi (ordine non importante per offset assoluti)
                for timestamp_key in group_obj.keys():
                    dataset = group_obj[timestamp_key][:]
                    if dataset.ndim == 2 and dataset.shape[0] > 0:
                        if dataset.shape[1] == 2:
                            # Gli offset sono ASSOLUTI - usa solo la colonna 0
                            all_times.extend(dataset[:, 0].tolist())
                            all_values.extend(dataset[:, 1].tolist())
            
            if len(all_times) > 0:
                df = pd.DataFrame({
                    'time': all_times,
                    'value': all_values
                })
                # Ordina per tempo
                df = df.sort_values('time').reset_index(drop=True)
                
                # Converti in datetime se richiesto
                if return_datetime:
                    # Trova il tempo minimo per calcolare offset relativi
                    min_time = df['time'].min()
                    df['time_relative'] = df['time'] - min_time
                    
                    # Usa la data di inizio acquisizione (dalla prima chiave timestamp)
                    # per convertire gli offset assoluti in datetime reali
                    if first_datetime is not None:
                        # Gli offset partono da ~10 secondi dopo l'inizio
                        # Calcola: data_inizio + (offset - offset_minimo)
                        df['datetime'] = first_datetime + pd.to_timedelta(df['time_relative'], unit='s')
                    else:
                        # Fallback: usa epoch (1970)
                        df['datetime'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Riordina colonne: datetime, time (assoluto), time_relative, value
                    df = df[['datetime', 'time', 'time_relative', 'value']]
                
                logger.debug(f"{signal_name}: {len(df)} campioni estratti")
                return df
            else:
                logger.warning(f"Nessun dato trovato per {signal_name}")
                return None
            
        except Exception as e:
            logger.warning(f"Errore nell'estrazione di {signal_name}: {e}")
            return None
    
    def load_signals(self, h5_path: Union[str, Path], return_datetime: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Carica tutti i segnali configurati dal file H5.
        
        Args:
            h5_path: Path al file H5
            return_datetime: Se True, include colonna datetime
            
        Returns:
            Dizionario {signal_name: DataFrame con colonne ['datetime', 'time', 'time_relative', 'value']}
        """
        h5_path = Path(h5_path)
        
        if not h5_path.exists():
            raise FileNotFoundError(f"File H5 non trovato: {h5_path}")
        
        logger.info(f"Caricamento segnali da: {h5_path.name}")
        
        signals_data = {}
        use_progress = self.config['logging']['progress_bars']
        
        with h5py.File(h5_path, 'r') as h5_file:
            iterator = self.signals_config.items()
            
            if use_progress:
                iterator = tqdm(iterator, desc="Estrazione segnali", leave=False)
            
            for signal_name, (group, name, subgroup, col) in iterator:
                df = self.extract_time_series_lazy(h5_file, group, name, subgroup, col, return_datetime)
                if df is not None:
                    signals_data[signal_name] = df
        
        logger.info(f"Estratti {len(signals_data)}/{len(self.signals_config)} segnali")
        
        if len(signals_data) == 0:
            raise ValueError("Nessun segnale estratto con successo dal file H5")
        
        return signals_data
    
    def align_signals(self, signals_data: Dict[str, pd.DataFrame], use_datetime: bool = True) -> pd.DataFrame:
        """
        Allinea temporalmente tutti i segnali tramite interpolazione.
        
        Usa il segnale con meno campioni come riferimento temporale e interpola
        tutti gli altri segnali su questa base temporale.
        
        Args:
            signals_data: Dizionario {signal_name: DataFrame}
            use_datetime: Se True, mantiene colonna datetime nel risultato
            
        Returns:
            DataFrame con tutti i segnali allineati temporalmente
        """
        if not signals_data:
            raise ValueError("signals_data è vuoto, impossibile allineare")
        
        # Trova il segnale con meno campioni (reference time)
        min_samples = min(len(df) for df in signals_data.values())
        reference_signal = min(signals_data.items(), key=lambda x: len(x[1]))
        reference_df = reference_signal[1]
        
        # Usa 'time' (assoluto) per interpolazione
        reference_time = reference_df['time'].values
        
        logger.info(f"Reference time: {len(reference_time)} campioni "
                   f"(da segnale '{reference_signal[0]}')")
        
        # Crea DataFrame allineato con datetime se disponibile
        if use_datetime and 'datetime' in reference_df.columns:
            aligned_df = pd.DataFrame({
                'datetime': reference_df['datetime'].values,
                'time': reference_time,
                'time_relative': reference_df['time_relative'].values
            })
        else:
            aligned_df = pd.DataFrame({'time': reference_time})
        
        # Parametri interpolazione da config
        interp_method = self.config['preprocessing']['interpolation']['method']
        fill_value = self.config['preprocessing']['interpolation']['fill_value']
        
        # Interpola tutti i segnali
        use_progress = self.config['logging']['progress_bars']
        iterator = signals_data.items()
        
        if use_progress:
            iterator = tqdm(iterator, desc="Interpolazione segnali", leave=False)
        
        for signal_name, df in iterator:
            try:
                interp_func = interp1d(
                    df['time'], 
                    df['value'],
                    kind=interp_method,
                    fill_value=fill_value
                )
                aligned_df[signal_name] = interp_func(reference_time)
                logger.debug(f"Interpolato: {signal_name}")
            except Exception as e:
                logger.error(f"Errore nell'interpolazione di {signal_name}: {e}")
                raise
        
        logger.info(f"Dataset allineato: {aligned_df.shape}")
        return aligned_df
    

    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        h5_path: Union[str, Path],
        output_format: str = 'csv'
    ) -> Path:
        """
        Salva il dataset processato nella directory di output configurata.
        
        Args:
            df: DataFrame da salvare
            h5_path: Path originale del file H5 (per naming)
            output_format: Formato output ('csv', 'parquet', 'pickle')
            
        Returns:
            Path al file salvato
        """
        # Directory output da config
        output_dir = Path(self.config['paths']['output_directory'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nome file basato su H5 originale
        filename = Path(h5_path).stem
        
        # Salva nel formato richiesto
        if output_format == 'csv':
            output_path = output_dir / f"{filename}_processed.csv"
            df.to_csv(output_path, index=False)
        elif output_format == 'parquet':
            output_path = output_dir / f"{filename}_processed.parquet"
            df.to_parquet(output_path, index=False)
        elif output_format == 'pickle':
            output_path = output_dir / f"{filename}_processed.pkl"
            df.to_pickle(output_path)
        else:
            raise ValueError(f"Formato non supportato: {output_format}")
        
        logger.info(f"Dataset salvato: {output_path}")
        return output_path


    def load_and_align(self, h5_path: Union[str, Path], return_datetime: bool = True, save_output=True) -> pd.DataFrame:
        """
        Pipeline completa: caricamento + allineamento.
        
        Args:
            h5_path: Path al file H5
            return_datetime: Se True, include colonna datetime
            
        Returns:
            DataFrame con tutti i segnali allineati temporalmente
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Caricamento file: {Path(h5_path).name}")
        logger.info(f"{'='*60}")
        
        # Step 1: Carica segnali
        signals_data = self.load_signals(h5_path, return_datetime)
        
        # Step 2: Allinea temporalmente
        aligned_df = self.align_signals(signals_data, use_datetime=return_datetime)
        
        if save_output:
            self.save_processed_data(aligned_df, h5_path)

        logger.info(f"Pipeline completata con successo")
        return aligned_df
    
    def load_multiple_files(
        self, 
        pattern: Optional[str] = None,
        base_dir: Optional[Union[str, Path]] = None
    ) -> List[Tuple[str, pd.DataFrame]]:
        """
        Carica multipli file H5 seguendo un pattern.
        
        Args:
            pattern: Pattern glob per ricerca file (default da config)
            base_dir: Directory base per ricerca (default da config)
            
        Returns:
            Lista di tuple (filename, dataframe)
        """
        if base_dir is None:
            base_dir = Path(self.config['paths']['data_directory'])
        else:
            base_dir = Path(base_dir)
        
        if pattern is None:
            pattern = self.config['paths']['h5_pattern']
        
        # Trova tutti i file H5
        h5_files = list(base_dir.glob(pattern))
        
        if not h5_files:
            logger.warning(f"Nessun file H5 trovato con pattern '{pattern}' in {base_dir}")
            return []
        
        logger.info(f"Trovati {len(h5_files)} file H5")
        
        results = []
        continue_on_error = self.config['batch_processing']['continue_on_error']
        
        for h5_file in tqdm(h5_files, desc="Batch processing"):
            try:
                df = self.load_and_align(h5_file)
                results.append((h5_file.stem, df))
                logger.info(f"{h5_file.name}: {df.shape}")
            except Exception as e:
                logger.error(f"Errore con {h5_file.name}: {e}")
                if not continue_on_error:
                    raise
                continue
        
        logger.info(f"\nBatch processing completato: {len(results)}/{len(h5_files)} file")
        return results
    
    
    def get_signal_info(self) -> pd.DataFrame:
        """
        Restituisce informazioni sui segnali configurati.
        
        Returns:
            DataFrame con informazioni sui segnali
        """
        info_list = []
        
        for category in ['vibration', 'temperature', 'generic', 'fieldbus']:
            if category in self.config['signals']:
                for signal in self.config['signals'][category]:
                    info_list.append({
                        'category': category,
                        'name': signal['name'],
                        'group': signal['group'],
                        'sensor': signal['sensor'],
                        'subgroup': signal['subgroup'],
                        'column': signal['column'],
                        'description': signal.get('description', 'N/A')
                    })
        
        return pd.DataFrame(info_list)


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Utility function per caricare solo la configurazione.
    
    Args:
        config_path: Path al file YAML
        
    Returns:
        Dizionario con configurazione
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Test del data loader
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <config_path> [h5_file_path]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Inizializza loader
    loader = BonfiglioliDataLoader(config_path)
    
    # Mostra info segnali
    print("\nSegnali configurati:")
    print(loader.get_signal_info().to_string(index=False))
    
    # Se fornito, carica un file H5
    if len(sys.argv) >= 3:
        h5_path = sys.argv[2]
        print(f"\nCaricamento file: {h5_path}")
        
        df = loader.load_and_align(h5_path)
        
        print(f"\nDataset caricato:")
        print(f"   Shape: {df.shape}")
        print(f"   Colonne: {list(df.columns)}")
        if 'datetime' in df.columns:
            print(f"   Periodo: {df['datetime'].min()} → {df['datetime'].max()}")
            print(f"   Durata: {(df['time'].max() - df['time'].min()) / 3600:.2f} ore")
        else:
            print(f"   Durata: {(df['time'].max() - df['time'].min()) / 3600:.2f} ore")
        print(f"\n{df.head()}")

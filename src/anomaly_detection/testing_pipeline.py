"""
Pipeline di Testing per Anomaly Detection Cross-Domain

Struttura:
1. Carica dati (Caratterizzazione, Fatica, Tampieri)
2. Per ogni combinazione (Modello × Domain Adaptation):
   - Applica DA
   - Allena su dati normali
   - Testa su Fatica (deve rilevare anomalie)
   - Testa su Tampieri (non deve dare falsi positivi)
3. Genera report comparativo

Uso:
    python testing_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import warnings
import json
from datetime import datetime

# Sklearn
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    QuantileTransformer, PowerTransformer
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from scipy import linalg

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

@dataclass
class Config:
    """Configurazione pipeline."""
    # Paths
    lab_caratterizzazione: str = "data/processed/ETR_946 - Prova 30_0 - 310 - Caratterizzazione 24h_1.csv"
    lab_fatica_dir: str = "data/processed"
    campo_tampieri: str = "data/tampieri/processed_streaming_row_continuous.csv"
    output_dir: str = "results/testing_pipeline"
    
    # Feature mapping
    lab_features: List[str] = field(default_factory=lambda: [
        'vib_rid_marc2_rms', 'vib_rid_marcia_rms', 
        'temp_mot_marcia', 'temp_cassa_riduttore'
    ])
    campo_features: List[str] = field(default_factory=lambda: [
        'bonfi/gb1_p4_acc_rms', 'bonfi/gb1_p3_acc_rms',
        'bonfi/gb1_p4_temp', 'bonfi/gb1_p3_temp'
    ])
    unified_features: List[str] = field(default_factory=lambda: [
        'vib_1', 'vib_2', 'temp_1', 'temp_2'
    ])
    
    # Sampling
    n_samples_train: int = 50000
    n_samples_test: int = 25000
    random_state: int = 42
    
    # Model params
    contamination: float = 0.05
    
    # Training strategy: 'lab_only', 'lab_and_campo', 'campo_only'
    training_strategy: str = 'lab_only'


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """Carica e prepara i dati."""
    
    def __init__(self, config: Config):
        self.config = config
        self.lab_to_unified = dict(zip(config.lab_features, config.unified_features))
        self.campo_to_unified = dict(zip(config.campo_features, config.unified_features))
    
    def _load_and_rename(self, path: str, mapping: dict) -> pd.DataFrame:
        """Carica CSV e rinomina colonne."""
        df = pd.read_csv(path)
        df_out = pd.DataFrame()
        for old, new in mapping.items():
            if old in df.columns:
                df_out[new] = df[old].values
        return df_out.dropna()
    
    def load_caratterizzazione(self) -> pd.DataFrame:
        """Carica dati Lab normali."""
        return self._load_and_rename(
            self.config.lab_caratterizzazione, 
            self.lab_to_unified
        )
    
    def load_fatica(self, take_last_fraction: float = 0.3) -> pd.DataFrame:
        """Carica dati Lab Fatica (anomali)."""
        fatica_files = sorted(Path(self.config.lab_fatica_dir).glob("*Fatica*.csv"))
        dfs = []
        for f in fatica_files:
            df = self._load_and_rename(str(f), self.lab_to_unified)
            dfs.append(df)
        
        df_all = pd.concat(dfs, ignore_index=True)
        
        # Prendi ultima parte (degrado avanzato)
        n = int(len(df_all) * take_last_fraction)
        return df_all.tail(n)
    
    def load_tampieri(self) -> pd.DataFrame:
        """Carica dati Campo (normali)."""
        return self._load_and_rename(
            self.config.campo_tampieri,
            self.campo_to_unified
        )
    
    def get_datasets(self) -> Dict[str, np.ndarray]:
        """Carica tutti i dataset."""
        cfg = self.config
        
        # Caratterizzazione (train - normale Lab)
        df_caratt = self.load_caratterizzazione()
        if len(df_caratt) > cfg.n_samples_train:
            df_caratt = df_caratt.sample(n=cfg.n_samples_train, random_state=cfg.random_state)
        
        # Fatica (test - anomalo)
        df_fatica = self.load_fatica()
        if len(df_fatica) > cfg.n_samples_test:
            df_fatica = df_fatica.sample(n=cfg.n_samples_test, random_state=cfg.random_state)
        
        # Tampieri (test - normale Campo)
        df_tampieri = self.load_tampieri()
        
        # Split Tampieri: parte per training, parte per test
        if cfg.training_strategy == 'lab_and_campo':
            n_tampieri_train = min(cfg.n_samples_train // 2, len(df_tampieri) // 2)
            n_tampieri_test = min(cfg.n_samples_test, len(df_tampieri) - n_tampieri_train)
            
            df_tampieri_shuffled = df_tampieri.sample(frac=1, random_state=cfg.random_state)
            df_tampieri_train = df_tampieri_shuffled.iloc[:n_tampieri_train]
            df_tampieri_test = df_tampieri_shuffled.iloc[n_tampieri_train:n_tampieri_train + n_tampieri_test]
            
            # Combina per training
            df_train = pd.concat([df_caratt, df_tampieri_train], ignore_index=True)
        elif cfg.training_strategy == 'campo_only':
            n_tampieri_train = min(cfg.n_samples_train, len(df_tampieri) // 2)
            n_tampieri_test = min(cfg.n_samples_test, len(df_tampieri) - n_tampieri_train)
            
            df_tampieri_shuffled = df_tampieri.sample(frac=1, random_state=cfg.random_state)
            df_tampieri_train = df_tampieri_shuffled.iloc[:n_tampieri_train]
            df_tampieri_test = df_tampieri_shuffled.iloc[n_tampieri_train:n_tampieri_train + n_tampieri_test]
            
            df_train = df_tampieri_train
        else:  # lab_only
            df_train = df_caratt
            if len(df_tampieri) > cfg.n_samples_test:
                df_tampieri_test = df_tampieri.sample(n=cfg.n_samples_test, random_state=cfg.random_state)
            else:
                df_tampieri_test = df_tampieri
        
        print(f"Dataset caricati (strategy: {cfg.training_strategy}):")
        print(f"  Train:                     {len(df_train):,} samples")
        print(f"  Fatica (test anomalo):     {len(df_fatica):,} samples")
        print(f"  Tampieri (test normale):   {len(df_tampieri_test):,} samples")
        
        return {
            'train': df_train.values,
            'test_anomaly': df_fatica.values,
            'test_normal': df_tampieri_test.values,
        }


# =============================================================================
# DOMAIN ADAPTATION METHODS
# =============================================================================

class DomainAdapter(ABC):
    """Classe base per Domain Adaptation."""
    
    name: str = "base"
    
    @abstractmethod
    def fit(self, X_source: np.ndarray) -> 'DomainAdapter':
        """Fit sui dati sorgente."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Trasforma i dati."""
        pass
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class NoAdapter(DomainAdapter):
    """Nessuna trasformazione (baseline)."""
    name = "None"
    
    def fit(self, X: np.ndarray) -> 'NoAdapter':
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.copy()


class StandardAdapter(DomainAdapter):
    """StandardScaler (z-score)."""
    name = "StandardScaler"
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray) -> 'StandardAdapter':
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


class RobustAdapter(DomainAdapter):
    """RobustScaler (mediana + IQR)."""
    name = "RobustScaler"
    
    def __init__(self):
        self.scaler = RobustScaler()
    
    def fit(self, X: np.ndarray) -> 'RobustAdapter':
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


class MinMaxAdapter(DomainAdapter):
    """MinMaxScaler [0, 1]."""
    name = "MinMaxScaler"
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def fit(self, X: np.ndarray) -> 'MinMaxAdapter':
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


class QuantileAdapter(DomainAdapter):
    """QuantileTransformer (distribuzione uniforme o normale)."""
    name = "QuantileTransform"
    
    def __init__(self, output_distribution: str = 'normal'):
        self.scaler = QuantileTransformer(
            output_distribution=output_distribution,
            random_state=42
        )
    
    def fit(self, X: np.ndarray) -> 'QuantileAdapter':
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


class CORALAdapter(DomainAdapter):
    """
    CORAL: CORrelation ALignment.
    Allinea la matrice di covarianza del target a quella del source.
    
    Paper: "Return of Frustratingly Easy Domain Adaptation" (Sun et al., 2016)
    """
    name = "CORAL"
    
    def __init__(self, regularization: float = 1e-6):
        self.reg = regularization
        self.source_mean = None
        self.source_cov_sqrt_inv = None
    
    def fit(self, X_source: np.ndarray) -> 'CORALAdapter':
        """Fit sulla sorgente: calcola mean e cov^(-1/2)."""
        self.source_mean = np.mean(X_source, axis=0)
        
        # Covarianza sorgente
        X_centered = X_source - self.source_mean
        cov_source = np.cov(X_centered.T) + self.reg * np.eye(X_source.shape[1])
        
        # Radice quadrata inversa della covarianza
        eigvals, eigvecs = linalg.eigh(cov_source)
        eigvals = np.maximum(eigvals, self.reg)
        self.source_cov_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        self.source_cov_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        return self
    
    def transform(self, X_target: np.ndarray) -> np.ndarray:
        """Trasforma il target per allinearlo alla sorgente."""
        # Centra rispetto alla media del target
        target_mean = np.mean(X_target, axis=0)
        X_centered = X_target - target_mean
        
        # Covarianza target
        cov_target = np.cov(X_centered.T) + self.reg * np.eye(X_target.shape[1])
        
        # Radice quadrata della covarianza target
        eigvals, eigvecs = linalg.eigh(cov_target)
        eigvals = np.maximum(eigvals, self.reg)
        target_cov_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        # Whitening + Coloring: X_aligned = X_centered @ Cov_t^(-1/2) @ Cov_s^(1/2)
        X_whitened = X_centered @ target_cov_sqrt_inv
        X_aligned = X_whitened @ self.source_cov_sqrt
        
        # Ricentra sulla media sorgente
        X_aligned += self.source_mean
        
        return X_aligned


# =============================================================================
# ANOMALY DETECTION MODELS
# =============================================================================

class AnomalyModel(ABC):
    """Classe base per modelli di anomaly detection."""
    
    name: str = "base"
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'AnomalyModel':
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ritorna -1 per anomalie, 1 per normali."""
        pass
    
    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Ritorna anomaly score (più alto = più anomalo)."""
        pass


class MahalanobisModel(AnomalyModel):
    """Distanza di Mahalanobis."""
    name = "Mahalanobis"
    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.mean = None
        self.cov_inv = None
        self.threshold = None
    
    def fit(self, X: np.ndarray) -> 'MahalanobisModel':
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
        self.cov_inv = np.linalg.inv(cov)
        
        scores = self.score(X)
        self.threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        diff = X - self.mean
        left = np.dot(diff, self.cov_inv)
        return np.sqrt(np.sum(left * diff, axis=1))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return np.where(scores > self.threshold, -1, 1)


class IsolationForestModel(AnomalyModel):
    """Isolation Forest."""
    name = "IsolationForest"
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
    
    def fit(self, X: np.ndarray) -> 'IsolationForestModel':
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


class OneClassSVMModel(AnomalyModel):
    """One-Class SVM."""
    name = "OneClassSVM"
    
    def __init__(self, nu: float = 0.05):
        self.model = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    
    def fit(self, X: np.ndarray) -> 'OneClassSVMModel':
        # Subsample per velocità
        if len(X) > 10000:
            idx = np.random.choice(len(X), 10000, replace=False)
            X = X[idx]
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


class LOFModel(AnomalyModel):
    """Local Outlier Factor."""
    name = "LOF"
    
    def __init__(self, contamination: float = 0.05):
        self.model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True
        )
    
    def fit(self, X: np.ndarray) -> 'LOFModel':
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


class EllipticModel(AnomalyModel):
    """Elliptic Envelope."""
    name = "EllipticEnvelope"
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray) -> 'EllipticModel':
        try:
            self.model.fit(X)
            self._fitted = True
        except Exception:
            self._fitted = False
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.ones(len(X))
        return self.model.predict(X)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X))
        return -self.model.decision_function(X)


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class EvalResult:
    """Risultato valutazione."""
    model: str
    adapter: str
    
    # Su dati anomali (Fatica)
    recall_anomaly: float      # % anomalie correttamente rilevate
    precision_anomaly: float   # % predizioni anomaly corrette
    
    # Su dati normali (Tampieri)  
    specificity_normal: float  # % normali correttamente classificati (1 - FPR)
    false_positive_rate: float # % normali erroneamente flaggati
    
    # Score
    f1_score: float
    balanced_accuracy: float


class Evaluator:
    """Valuta le predizioni."""
    
    @staticmethod
    def evaluate(y_pred_anomaly: np.ndarray, 
                 y_pred_normal: np.ndarray,
                 model_name: str,
                 adapter_name: str) -> EvalResult:
        """
        Valuta predizioni.
        
        Args:
            y_pred_anomaly: Predizioni su test anomalo (-1=anomaly, 1=normal)
            y_pred_normal: Predizioni su test normale (-1=anomaly, 1=normal)
        """
        # Su anomali: vogliamo predire -1
        tp = (y_pred_anomaly == -1).sum()
        fn = (y_pred_anomaly == 1).sum()
        
        # Su normali: vogliamo predire 1
        tn = (y_pred_normal == 1).sum()
        fp = (y_pred_normal == -1).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        balanced_acc = (recall + specificity) / 2
        
        return EvalResult(
            model=model_name,
            adapter=adapter_name,
            recall_anomaly=recall,
            precision_anomaly=precision,
            specificity_normal=specificity,
            false_positive_rate=fpr,
            f1_score=f1,
            balanced_accuracy=balanced_acc
        )


# =============================================================================
# PIPELINE
# =============================================================================

class TestingPipeline:
    """Pipeline di testing completa."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Definisci modelli e adapter
        self.adapters = [
            NoAdapter(),
            StandardAdapter(),
            RobustAdapter(),
            MinMaxAdapter(),
            QuantileAdapter(),
            CORALAdapter(),
        ]
        
        self.models = [
            MahalanobisModel(contamination=self.config.contamination),
            IsolationForestModel(contamination=self.config.contamination),
            OneClassSVMModel(nu=self.config.contamination),
            LOFModel(contamination=self.config.contamination),
            EllipticModel(contamination=self.config.contamination),
        ]
    
    def run(self) -> pd.DataFrame:
        """Esegue la pipeline completa."""
        print("="*70)
        print("TESTING PIPELINE: Model x Domain Adaptation")
        print("="*70)
        
        # 1. Carica dati
        loader = DataLoader(self.config)
        data = loader.get_datasets()
        
        X_train = data['train']
        X_test_anomaly = data['test_anomaly']
        X_test_normal = data['test_normal']
        
        # 2. Test tutte le combinazioni
        results = []
        
        for adapter in self.adapters:
            print(f"\n{'-'*50}")
            print(f"Domain Adaptation: {adapter.name}")
            print(f"{'-'*50}")
            
            # Applica adapter
            adapter_fitted = type(adapter)()  # Fresh instance
            if isinstance(adapter_fitted, CORALAdapter):
                # CORAL: fit su source (train), transform su target
                adapter_fitted.fit(X_train)
                X_train_adapted = adapter_fitted.transform(X_train)
                X_test_anomaly_adapted = adapter_fitted.transform(X_test_anomaly)
                X_test_normal_adapted = adapter_fitted.transform(X_test_normal)
            else:
                # Altri: fit su train, transform tutto
                X_train_adapted = adapter_fitted.fit_transform(X_train)
                X_test_anomaly_adapted = adapter_fitted.transform(X_test_anomaly)
                X_test_normal_adapted = adapter_fitted.transform(X_test_normal)
            
            for model_class in self.models:
                # Fresh model instance
                model = type(model_class)(
                    contamination=self.config.contamination
                ) if hasattr(model_class, 'contamination') else type(model_class)()
                
                # Setta contamination se disponibile
                if hasattr(model, 'contamination'):
                    model.contamination = self.config.contamination
                
                try:
                    # Train
                    model.fit(X_train_adapted)
                    
                    # Predict
                    y_pred_anomaly = model.predict(X_test_anomaly_adapted)
                    y_pred_normal = model.predict(X_test_normal_adapted)
                    
                    # Evaluate
                    result = Evaluator.evaluate(
                        y_pred_anomaly, y_pred_normal,
                        model.name, adapter_fitted.name
                    )
                    results.append(result)
                    
                    print(f"  {model.name:18} | Recall={result.recall_anomaly:.3f} | "
                          f"FPR={result.false_positive_rate:.3f} | F1={result.f1_score:.3f}")
                    
                except Exception as e:
                    print(f"  {model.name:18} | ERROR: {e}")
        
        # 3. Crea DataFrame risultati
        df_results = pd.DataFrame([
            {
                'Model': r.model,
                'DomainAdaptation': r.adapter,
                'Recall': r.recall_anomaly,
                'Precision': r.precision_anomaly,
                'Specificity': r.specificity_normal,
                'FPR': r.false_positive_rate,
                'F1': r.f1_score,
                'BalancedAcc': r.balanced_accuracy,
            }
            for r in results
        ])
        
        # 4. Salva e visualizza
        self._save_results(df_results)
        self._plot_results(df_results)
        
        # 5. Best combinations
        print("\n" + "="*70)
        print("TOP 5 COMBINATIONS (by F1 Score)")
        print("="*70)
        top5 = df_results.nlargest(5, 'F1')
        print(top5.to_string(index=False))
        
        return df_results
    
    def _save_results(self, df: pd.DataFrame):
        """Salva risultati."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV
        csv_path = self.output_dir / f"results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nRisultati salvati: {csv_path}")
        
        # JSON pivot
        pivot = df.pivot_table(
            index='Model', 
            columns='DomainAdaptation', 
            values=['Recall', 'FPR', 'F1']
        )
        json_path = self.output_dir / f"results_{timestamp}.json"
        pivot.to_json(json_path)
    
    def _plot_results(self, df: pd.DataFrame):
        """Genera plot comparativi."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        adapters = df['DomainAdaptation'].unique()
        models = df['Model'].unique()
        
        # 1. Recall per DA method
        ax = axes[0, 0]
        pivot_recall = df.pivot(index='Model', columns='DomainAdaptation', values='Recall')
        pivot_recall.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('Recall (Anomaly Detection)')
        ax.set_title('Recall by Model and Domain Adaptation')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='DA Method', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_ylim(0, 1.1)
        
        # 2. FPR per DA method
        ax = axes[0, 1]
        pivot_fpr = df.pivot(index='Model', columns='DomainAdaptation', values='FPR')
        pivot_fpr.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('False Positive Rate')
        ax.set_title('FPR by Model and Domain Adaptation (lower is better)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='DA Method', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_ylim(0, 1.1)
        
        # 3. F1 heatmap
        ax = axes[1, 0]
        pivot_f1 = df.pivot(index='Model', columns='DomainAdaptation', values='F1')
        im = ax.imshow(pivot_f1.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(adapters)))
        ax.set_xticklabels(adapters, rotation=45, ha='right')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(pivot_f1.index)
        ax.set_title('F1 Score Heatmap')
        plt.colorbar(im, ax=ax)
        
        # Annotate
        for i in range(len(pivot_f1.index)):
            for j in range(len(adapters)):
                val = pivot_f1.values[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9)
        
        # 4. Balanced Accuracy heatmap
        ax = axes[1, 1]
        pivot_ba = df.pivot(index='Model', columns='DomainAdaptation', values='BalancedAcc')
        im = ax.imshow(pivot_ba.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(adapters)))
        ax.set_xticklabels(adapters, rotation=45, ha='right')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(pivot_ba.index)
        ax.set_title('Balanced Accuracy Heatmap')
        plt.colorbar(im, ax=ax)
        
        for i in range(len(pivot_ba.index)):
            for j in range(len(adapters)):
                val = pivot_ba.values[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9)
        
        plt.suptitle('Anomaly Detection: Model × Domain Adaptation Comparison', fontsize=14)
        plt.tight_layout()
        
        plot_path = self.output_dir / 'comparison_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot salvato: {plot_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Training strategy from command line
    strategy = sys.argv[1] if len(sys.argv) > 1 else 'lab_and_campo'
    
    print(f"\n{'#'*70}")
    print(f"# RUNNING WITH STRATEGY: {strategy}")
    print(f"{'#'*70}\n")
    
    config = Config(training_strategy=strategy)
    pipeline = TestingPipeline(config)
    results = pipeline.run()

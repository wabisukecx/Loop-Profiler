"""
Loop Profiler - Feature Extractor
音響特徴量抽出クラス（librosa使用）
"""

from pathlib import Path
from typing import Dict, Optional
import hashlib
import json

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa not found. Feature extraction will be disabled.")
    print("   Install: pip install librosa")


class FeatureExtractor:
    """
    音響特徴量抽出クラス
    
    Examples:
        >>> extractor = FeatureExtractor(cache_dir="LooperOutput/features_cache")
        >>> features = extractor.extract(
        ...     audio_path="song.mp3",
        ...     loop_start=264600,
        ...     loop_end=1323000,
        ...     sample_rate=44100
        ... )
        >>> print(features["amplitude_smoothness"])
        0.87
    """
    
    def __init__(self, cache_dir: str = "LooperOutput/features_cache"):
        """
        初期化
        
        Args:
            cache_dir: 特徴量キャッシュディレクトリ
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa is required for feature extraction.\n"
                "Install: pip install librosa"
            )
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract(
        self,
        audio_path: str,
        loop_start: int,
        loop_end: int,
        sample_rate: int,
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        ループ候補の音響特徴量を抽出
        
        Args:
            audio_path: 音声ファイルパス
            loop_start: ループ開始サンプル位置
            loop_end: ループ終了サンプル位置
            sample_rate: サンプリングレート
            use_cache: キャッシュを使用するか
        
        Returns:
            特徴量辞書 {
                "amplitude_smoothness": float (0-1),
                "spectral_similarity": float (0-1),
                "tempo_consistency": float (0-1),
                "loudness_matching": float (0-1)
            }
        """
        # キャッシュチェック
        if use_cache:
            cached = self._load_cache(audio_path, loop_start, loop_end)
            if cached is not None:
                return cached
        
        # 音声読み込み
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # 特徴量抽出
        features = {}
        
        # 1. Amplitude Smoothness（振幅の滑らかさ）
        features["amplitude_smoothness"] = self._calc_amplitude_smoothness(
            y, loop_start, loop_end, sr
        )
        
        # 2. Spectral Similarity（周波数成分の類似度）
        features["spectral_similarity"] = self._calc_spectral_similarity(
            y, loop_start, loop_end, sr
        )
        
        # 3. Tempo Consistency（BPM安定性）
        features["tempo_consistency"] = self._calc_tempo_consistency(
            y, loop_start, loop_end, sr
        )
        
        # 4. Loudness Matching（音圧バランス）
        features["loudness_matching"] = self._calc_loudness_matching(
            y, loop_start, loop_end, sr
        )
        
        # キャッシュ保存
        if use_cache:
            self._save_cache(audio_path, loop_start, loop_end, features)
        
        return features
    
    # ===== Feature Calculation Methods =====
    
    def _calc_amplitude_smoothness(self, y: np.ndarray, start: int, 
                                    end: int, sr: int) -> float:
        """
        振幅の滑らかさを計算
        
        ループ境界±200msの振幅差が小さいほど高スコア
        """
        boundary_samples = int(sr * 0.2)  # 200ms
        
        # ループ開始付近
        start_chunk = y[max(0, start-boundary_samples):start+boundary_samples]
        # ループ終了付近
        end_chunk = y[max(0, end-boundary_samples):min(len(y), end+boundary_samples)]
        
        if len(start_chunk) == 0 or len(end_chunk) == 0:
            return 0.5  # デフォルト値
        
        # 平均振幅の差
        amp_diff = abs(np.mean(np.abs(start_chunk)) - np.mean(np.abs(end_chunk)))
        
        # 0-1に正規化（差が小さいほど1に近い）
        smoothness = max(0.0, 1.0 - amp_diff * 10)
        
        return float(smoothness)
    
    def _calc_spectral_similarity(self, y: np.ndarray, start: int,
                                   end: int, sr: int) -> float:
        """
        周波数成分の類似度を計算（MFCC相関）
        
        ループ境界の周波数特性が似ているほど高スコア
        """
        boundary_samples = int(sr * 0.2)
        
        start_chunk = y[max(0, start-boundary_samples):start+boundary_samples]
        end_chunk = y[max(0, end-boundary_samples):min(len(y), end+boundary_samples)]
        
        if len(start_chunk) < 512 or len(end_chunk) < 512:
            return 0.5  # サンプル不足時はデフォルト値
        
        try:
            # MFCC抽出
            mfcc_start = librosa.feature.mfcc(y=start_chunk, sr=sr, n_mfcc=13)
            mfcc_end = librosa.feature.mfcc(y=end_chunk, sr=sr, n_mfcc=13)
            
            # 相関係数計算
            corr_matrix = np.corrcoef(mfcc_start.flatten(), mfcc_end.flatten())
            correlation = corr_matrix[0, 1]
            
            # NaN対策
            if np.isnan(correlation):
                return 0.5
            
            # -1～1を0～1に正規化
            similarity = (correlation + 1.0) / 2.0
            
            return float(max(0.0, min(1.0, similarity)))
        
        except Exception as e:
            print(f"⚠️ MFCC計算エラー: {e}")
            return 0.5
    
    def _calc_tempo_consistency(self, y: np.ndarray, start: int,
                                 end: int, sr: int) -> float:
        """
        BPM安定性を計算
        
        ループ区間のテンポが一定であるほど高スコア
        """
        loop_section = y[start:end]
        
        if len(loop_section) < sr * 2:  # 最低2秒必要
            return 0.5
        
        try:
            # ビート検出
            tempo, beats = librosa.beat.beat_track(y=loop_section, sr=sr)
            
            if len(beats) < 2:
                return 0.5  # ビート検出失敗時は中立値
            
            # ビート間隔の標準偏差
            beat_intervals = np.diff(beats)
            tempo_variance = np.std(beat_intervals)
            
            # 分散が小さいほど高スコア
            consistency = 1.0 / (1.0 + tempo_variance / 10.0)
            
            return float(max(0.0, min(1.0, consistency)))
        
        except Exception as e:
            print(f"⚠️ Tempo分析エラー: {e}")
            return 0.5
    
    def _calc_loudness_matching(self, y: np.ndarray, start: int,
                                 end: int, sr: int) -> float:
        """
        音圧バランスを計算（RMS差）
        
        ループ境界の音圧レベルが揃っているほど高スコア
        """
        boundary_samples = int(sr * 0.2)
        
        start_chunk = y[max(0, start-boundary_samples):start+boundary_samples]
        end_chunk = y[max(0, end-boundary_samples):min(len(y), end+boundary_samples)]
        
        if len(start_chunk) == 0 or len(end_chunk) == 0:
            return 0.5
        
        try:
            # RMS計算
            rms_start = librosa.feature.rms(y=start_chunk)[0]
            rms_end = librosa.feature.rms(y=end_chunk)[0]
            
            # RMS差
            rms_diff = abs(np.mean(rms_start) - np.mean(rms_end))
            
            # 0-1に正規化
            matching = max(0.0, 1.0 - rms_diff * 20)
            
            return float(matching)
        
        except Exception as e:
            print(f"⚠️ RMS計算エラー: {e}")
            return 0.5
    
    # ===== Cache Management =====
    
    def _load_cache(self, audio_path: str, start: int, end: int) -> Optional[Dict]:
        """キャッシュから特徴量を読み込み"""
        cache_key = self._get_cache_key(audio_path, start, end)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except:
                pass
        
        return None
    
    def _save_cache(self, audio_path: str, start: int, end: int, 
                    features: Dict):
        """特徴量をキャッシュに保存"""
        cache_key = self._get_cache_key(audio_path, start, end)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_file.write_text(json.dumps(features, indent=2))
    
    @staticmethod
    def _get_cache_key(audio_path: str, start: int, end: int) -> str:
        """キャッシュキーを生成"""
        filename = Path(audio_path).name
        key_str = f"{filename}_{start}_{end}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]


if __name__ == "__main__":
    # テスト実行
    if LIBROSA_AVAILABLE:
        print("=== FeatureExtractor Test ===")
        print("⚠️ テストには実際の音声ファイルが必要です")
        print("   Example:")
        print("   extractor = FeatureExtractor()")
        print("   features = extractor.extract('song.mp3', 264600, 1323000, 44100)")
    else:
        print("❌ librosa not available. Cannot run test.")

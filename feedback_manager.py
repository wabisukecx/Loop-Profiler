"""
Loop Profiler - Feedback Manager
フィードバックデータの管理クラス（CRUD操作、永続化、統計計算）
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import hashlib
from datetime import datetime
import sys
import platform

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ numpy not found. Some features may be limited.")


class FeedbackManager:
    """
    ループフィードバックデータの管理クラス
    
    Examples:
        >>> manager = FeedbackManager("LooperOutput/feedback.json")
        >>> manager.add_feedback(
        ...     audio_path="song.mp3",
        ...     loop_start=264600,
        ...     loop_end=1323000,
        ...     features={"amplitude_smoothness": 0.87, ...},
        ...     rating=1,
        ...     pymusiclooper_score=0.956
        ... )
        >>> X, y = manager.get_training_data()
    """
    
    def __init__(self, filepath: str = "LooperOutput/feedback.json"):
        """
        初期化
        
        Args:
            filepath: フィードバックJSONファイルのパス
        """
        self.filepath = Path(filepath)
        self.data = self._load()
        self._cache = {}  # audio_hash -> feedbacks のキャッシュ
    
    # ===== Core CRUD Operations =====
    
    def add_feedback(
        self, 
        audio_path: str,
        loop_start: int,
        loop_end: int,
        features: Dict[str, float],
        rating: int,
        pymusiclooper_score: float,
        audio_metadata: Optional[Dict] = None,
        exported: bool = False,
        export_settings: Optional[Dict] = None,
        source: str = "manual"
    ) -> int:
        """
        新規フィードバックを追加
        
        Args:
            audio_path: 音声ファイルパス
            loop_start: ループ開始サンプル位置
            loop_end: ループ終了サンプル位置
            features: 特徴量辞書
            rating: ユーザー評価 (1=good, 0=bad)
            pymusiclooper_score: Pymusiclooperスコア (0.0-1.0)
            audio_metadata: 音声メタデータ (optional)
            exported: エクスポート済みフラグ
            export_settings: エクスポート設定 (optional)
            source: フィードバック元 ("thumbs_up"|"thumbs_down"|"export"|"manual")
        
        Returns:
            追加されたフィードバックのID
        """
        audio_hash = self._hash_file(audio_path)
        audio_name = Path(audio_path).name
        
        # メタデータのデフォルト値
        if audio_metadata is None:
            audio_metadata = self._get_default_metadata()
        
        # ループ候補情報の計算
        sample_rate = audio_metadata.get("sample_rate", 44100)
        loop_candidate = self._create_loop_candidate_info(
            loop_start, loop_end, sample_rate
        )
        
        # フィードバックレコード作成
        feedback_id = len(self.data["feedbacks"]) + 1
        now = datetime.now().isoformat()
        
        feedback = {
            "id": feedback_id,
            "audio_file": audio_name,
            "audio_hash": audio_hash,
            "audio_metadata": audio_metadata,
            "loop_candidate": loop_candidate,
            "scores": {
                "pymusiclooper": pymusiclooper_score,
                "ai_predicted": None,
                "ai_confidence": None
            },
            "features": features,
            "user_feedback": {
                "rating": rating,
                "explicit": source in ["thumbs_up", "thumbs_down"],
                "source": source
            },
            "export_info": {
                "exported": exported,
                "export_count": 1 if exported else 0,
                "last_export_at": now if exported else None,
                "export_settings": export_settings
            },
            "timestamps": {
                "created_at": now,
                "updated_at": now
            }
        }
        
        self.data["feedbacks"].append(feedback)
        self._update_statistics(rating, exported)
        self.save()
        self._cache.clear()
        
        return feedback_id
    
    def update_ai_score(self, feedback_id: int, ai_score: float, 
                        confidence: float) -> bool:
        """
        AI予測スコアを更新
        
        Args:
            feedback_id: フィードバックID
            ai_score: AI予測スコア (0-100)
            confidence: 予測信頼度 (0.0-1.0)
        
        Returns:
            更新成功時True
        """
        for fb in self.data["feedbacks"]:
            if fb["id"] == feedback_id:
                fb["scores"]["ai_predicted"] = ai_score
                fb["scores"]["ai_confidence"] = confidence
                fb["timestamps"]["updated_at"] = datetime.now().isoformat()
                self.save()
                return True
        return False
    
    def update_export_info(self, feedback_id: int, 
                          export_settings: Dict) -> bool:
        """
        エクスポート情報を更新
        
        Args:
            feedback_id: フィードバックID
            export_settings: エクスポート設定
        
        Returns:
            更新成功時True
        """
        for fb in self.data["feedbacks"]:
            if fb["id"] == feedback_id:
                export_info = fb["export_info"]
                export_info["exported"] = True
                export_info["export_count"] += 1
                export_info["last_export_at"] = datetime.now().isoformat()
                export_info["export_settings"] = export_settings
                
                fb["timestamps"]["updated_at"] = datetime.now().isoformat()
                
                # 統計更新
                if export_info["export_count"] == 1:
                    self.data["statistics"]["exported_count"] += 1
                self.data["statistics"]["total_export_operations"] += 1
                
                self.save()
                return True
        return False
    
    def delete_feedback(self, feedback_id: int) -> bool:
        """
        フィードバックを削除
        
        Args:
            feedback_id: 削除するフィードバックID
        
        Returns:
            削除成功時True
        """
        for i, fb in enumerate(self.data["feedbacks"]):
            if fb["id"] == feedback_id:
                removed = self.data["feedbacks"].pop(i)
                
                # 統計更新
                rating = removed["user_feedback"]["rating"]
                if rating == 1:
                    self.data["statistics"]["positive_count"] -= 1
                else:
                    self.data["statistics"]["negative_count"] -= 1
                
                self.data["statistics"]["total_feedbacks"] -= 1
                
                self.save()
                self._cache.clear()
                return True
        return False
    
    # ===== Query Operations =====
    
    def get_by_audio(self, audio_path: str) -> List[Dict]:
        """
        特定の曲のフィードバックを取得（キャッシュ利用）
        
        Args:
            audio_path: 音声ファイルパス
        
        Returns:
            フィードバックのリスト
        """
        audio_hash = self._hash_file(audio_path)
        
        if audio_hash not in self._cache:
            self._cache[audio_hash] = [
                fb for fb in self.data["feedbacks"]
                if fb["audio_hash"] == audio_hash
            ]
        
        return self._cache[audio_hash]
    
    def get_by_id(self, feedback_id: int) -> Optional[Dict]:
        """
        IDでフィードバックを取得
        
        Args:
            feedback_id: フィードバックID
        
        Returns:
            フィードバック辞書、見つからない場合None
        """
        for fb in self.data["feedbacks"]:
            if fb["id"] == feedback_id:
                return fb
        return None
    
    def get_training_data(self) -> Tuple:
        """
        ML学習用のデータセットを取得
        
        Returns:
            (X, y) - 特徴量配列とラベル配列
            numpyが利用可能な場合: (np.ndarray, np.ndarray)
            numpyが利用不可の場合: (list, list)
        """
        X, y = [], []
        
        for fb in self.data["feedbacks"]:
            features = fb["features"]
            X.append([
                fb["scores"]["pymusiclooper"],
                features.get("amplitude_smoothness", 0.0),
                features.get("spectral_similarity", 0.0),
                features.get("tempo_consistency", 0.0),
                features.get("loudness_matching", 0.0)
            ])
            y.append(fb["user_feedback"]["rating"])
        
        if NUMPY_AVAILABLE:
            import numpy as np
            return np.array(X), np.array(y)
        else:
            return X, y
    
    def get_good_loops(self, min_ai_score: float = 80.0) -> List[Dict]:
        """
        高評価のループを取得
        
        Args:
            min_ai_score: 最低AIスコア閾値
        
        Returns:
            条件を満たすフィードバックのリスト
        """
        return [
            fb for fb in self.data["feedbacks"]
            if fb["user_feedback"]["rating"] == 1
            and fb["scores"].get("ai_predicted", 0) >= min_ai_score
        ]
    
    # ===== Statistics =====
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        return self.data["statistics"].copy()
    
    def update_model_performance(self, accuracy: float, precision: float,
                                  recall: float, n_samples: int):
        """
        モデル性能指標を更新
        
        Args:
            accuracy: 正解率
            precision: 適合率
            recall: 再現率
            n_samples: 学習サンプル数
        """
        self.data["statistics"]["model_performance"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "training_samples": n_samples
        }
        self.data["statistics"]["last_model_training"] = datetime.now().isoformat()
        self.save()
    
    # ===== Persistence =====
    
    def save(self):
        """JSONファイルに保存"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
    
    def _load(self) -> Dict:
        """JSONファイルを読み込み"""
        if self.filepath.exists():
            try:
                return json.loads(self.filepath.read_text(encoding='utf-8'))
            except json.JSONDecodeError as e:
                print(f"⚠️ feedback.json読み込みエラー: {e}")
                # バックアップ作成
                backup = self.filepath.with_suffix('.json.backup')
                import shutil
                shutil.copy(self.filepath, backup)
                print(f"   バックアップ: {backup}")
        
        return self._create_empty_data()
    
    # ===== Private Helpers =====
    
    def _create_empty_data(self) -> Dict:
        """初期データ構造を生成"""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "app_version": "2.0.0",
            "feedbacks": [],
            "statistics": {
                "total_feedbacks": 0,
                "positive_count": 0,
                "negative_count": 0,
                "exported_count": 0,
                "total_export_operations": 0,
                "last_model_training": None,
                "model_performance": None
            },
            "metadata": {
                "system_info": self._get_system_info()
            }
        }
    
    def _hash_file(self, filepath: str) -> str:
        """ファイル名からハッシュ生成（MD5の最初16文字）"""
        filename = Path(filepath).name
        full_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()
        return full_hash[:16]
    
    def _create_loop_candidate_info(self, start: int, end: int, 
                                     sr: int) -> Dict:
        """ループ候補情報を生成"""
        start_ms = (start * 1000) // sr
        end_ms = (end * 1000) // sr
        
        return {
            "start_sample": start,
            "end_sample": end,
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "loop_duration_ms": end_ms - start_ms
        }
    
    def _get_default_metadata(self) -> Dict:
        """デフォルトメタデータ"""
        return {
            "duration_ms": 0,
            "sample_rate": 44100,
            "channels": 2,
            "bitrate_kbps": None
        }
    
    def _update_statistics(self, rating: int, exported: bool):
        """統計を更新"""
        stats = self.data["statistics"]
        stats["total_feedbacks"] += 1
        
        if rating == 1:
            stats["positive_count"] += 1
        else:
            stats["negative_count"] += 1
        
        if exported:
            stats["exported_count"] += 1
            stats["total_export_operations"] += 1
    
    @staticmethod
    def _get_system_info() -> Dict:
        """システム情報を取得"""
        try:
            import librosa
            librosa_ver = librosa.__version__
        except:
            librosa_ver = "N/A"
        
        try:
            import sklearn
            sklearn_ver = sklearn.__version__
        except:
            sklearn_ver = "N/A"
        
        return {
            "os": platform.system() + " " + platform.release(),
            "python_version": sys.version.split()[0],
            "librosa_version": librosa_ver,
            "sklearn_version": sklearn_ver
        }


if __name__ == "__main__":
    # テスト実行
    print("=== FeedbackManager Test ===")
    manager = FeedbackManager("test_feedback.json")
    
    # フィードバック追加
    fb_id = manager.add_feedback(
        audio_path="test_song.mp3",
        loop_start=264600,
        loop_end=1323000,
        features={
            "amplitude_smoothness": 0.87,
            "spectral_similarity": 0.92,
            "tempo_consistency": 0.78,
            "loudness_matching": 0.89
        },
        rating=1,
        pymusiclooper_score=0.956
    )
    
    print(f"✅ Feedback added: ID={fb_id}")
    print(f"✅ Statistics: {manager.get_statistics()}")

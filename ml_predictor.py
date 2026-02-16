"""
Loop Profiler - ML Predictor
機械学習による品質予測クラス
"""

import pickle
from pathlib import Path
from typing import Tuple, Optional

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not found. ML prediction will be disabled.")
    print("   Install: pip install scikit-learn")


class MLPredictor:
    """
    ML品質予測クラス
    
    Examples:
        >>> predictor = MLPredictor("LooperOutput/loop_model.pkl")
        >>> X, y = feedback_manager.get_training_data()
        >>> predictor.train(X, y)
        >>> ai_score, confidence = predictor.predict([0.95, 0.87, 0.92, 0.78, 0.89])
        >>> print(f"AI Score: {ai_score:.1f}%")
        AI Score: 92.3%
    """
    
    MIN_TRAINING_SAMPLES = 10  # 最低学習サンプル数
    
    def __init__(self, model_path: str = "LooperOutput/loop_model.pkl"):
        """
        初期化
        
        Args:
            model_path: モデルファイルのパス
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for ML prediction.\n"
                "Install: pip install scikit-learn"
            )
        
        self.model_path = Path(model_path)
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.is_trained = False
        self._load_model()
    
    def train(self, X, y) -> bool:
        """
        モデルを学習
        
        Args:
            X: 特徴量配列 shape (N, 5) or list of lists
            y: ラベル配列 shape (N,) or list
        
        Returns:
            学習成功時True、サンプル不足時False
        """
        # numpyに変換（リストの場合）
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if len(X) < self.MIN_TRAINING_SAMPLES:
            print(f"⚠️ 学習データ不足: {len(X)}件 (最低{self.MIN_TRAINING_SAMPLES}件必要)")
            return False
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            self._save_model()
            print(f"✅ モデル学習完了: {len(X)}件")
            return True
        
        except Exception as e:
            print(f"❌ 学習エラー: {e}")
            return False
    
    def predict(self, features) -> Tuple[Optional[float], Optional[float]]:
        """
        AI品質スコアを予測
        
        Args:
            features: 特徴量リスト [pymusiclooper, amp, spec, tempo, loud]
        
        Returns:
            (ai_score, confidence) - AIスコア(0-100)と信頼度(0-1)
            学習前の場合は (None, None)
        """
        if not self.is_trained:
            return None, None
        
        try:
            # numpyに変換（リストの場合）
            if not isinstance(features, np.ndarray):
                features = np.array([features])
            else:
                features = features.reshape(1, -1)
            
            # 確率予測
            proba = self.model.predict_proba(features)[0]
            
            # good loopである確率
            ai_score = proba[1] * 100
            
            # 信頼度（確率の最大値）
            confidence = max(proba)
            
            return float(ai_score), float(confidence)
        
        except Exception as e:
            print(f"❌ 予測エラー: {e}")
            return None, None
    
    def evaluate(self, X, y) -> dict:
        """
        モデルを評価（クロスバリデーション）
        
        Args:
            X: 特徴量配列
            y: ラベル配列
        
        Returns:
            評価指標辞書 {accuracy, precision, recall, ...}
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        # numpyに変換
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if len(X) < 5:
            return {"error": "Not enough samples for evaluation"}
        
        try:
            # クロスバリデーション（5-fold）
            cv_folds = min(5, len(X))
            cv_scores = cross_val_score(self.model, X, y, cv=cv_folds)
            
            # 全データで予測
            y_pred = self.model.predict(X)
            
            return {
                "accuracy": float(np.mean(cv_scores)),
                "accuracy_std": float(np.std(cv_scores)),
                "precision": float(precision_score(y, y_pred, zero_division=0)),
                "recall": float(recall_score(y, y_pred, zero_division=0)),
                "n_samples": len(X)
            }
        
        except Exception as e:
            print(f"❌ 評価エラー: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        特徴量の重要度を取得
        
        Returns:
            重要度配列 [pymusiclooper, amp, spec, tempo, loud]
        """
        if not self.is_trained:
            return None
        
        return self.model.feature_importances_
    
    # ===== Model Persistence =====
    
    def _save_model(self):
        """モデルをファイルに保存"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def _load_model(self):
        """モデルをファイルから読み込み"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                print(f"✅ モデル読み込み完了: {self.model_path.name}")
            except Exception as e:
                print(f"⚠️ モデル読み込みエラー: {e}")
                self.is_trained = False


if __name__ == "__main__":
    # テスト実行
    if SKLEARN_AVAILABLE:
        print("=== MLPredictor Test ===")
        
        # ダミーデータで学習テスト
        import numpy as np
        
        predictor = MLPredictor("test_model.pkl")
        
        # 12件の学習データ（10件good, 2件bad）
        X_train = []
        y_train = []
        
        for i in range(12):
            rating = 1 if i < 10 else 0
            features = [
                0.95 if rating == 1 else 0.60,  # pymusiclooper
                0.85 if rating == 1 else 0.45,  # amplitude
                0.90 if rating == 1 else 0.50,  # spectral
                0.80 if rating == 1 else 0.40,  # tempo
                0.88 if rating == 1 else 0.48   # loudness
            ]
            X_train.append(features)
            y_train.append(rating)
        
        # 学習
        success = predictor.train(X_train, y_train)
        print(f"学習結果: {'成功' if success else '失敗'}")
        
        if success:
            # 予測テスト
            good_loop = [0.95, 0.87, 0.92, 0.78, 0.89]
            bad_loop = [0.60, 0.45, 0.50, 0.40, 0.48]
            
            ai_score_good, conf_good = predictor.predict(good_loop)
            ai_score_bad, conf_bad = predictor.predict(bad_loop)
            
            print(f"\n予測結果:")
            print(f"  Good Loop: AI Score={ai_score_good:.1f}%, Confidence={conf_good:.2f}")
            print(f"  Bad Loop:  AI Score={ai_score_bad:.1f}%, Confidence={conf_bad:.2f}")
            
            # 評価
            eval_result = predictor.evaluate(X_train, y_train)
            print(f"\nモデル評価:")
            print(f"  Accuracy:  {eval_result.get('accuracy', 0):.3f}")
            print(f"  Precision: {eval_result.get('precision', 0):.3f}")
            print(f"  Recall:    {eval_result.get('recall', 0):.3f}")
    else:
        print("❌ scikit-learn not available. Cannot run test.")

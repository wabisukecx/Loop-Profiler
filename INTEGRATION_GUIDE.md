# Loop Profiler v2.0 - AI機能統合手順書

## 📋 統合手順（ステップバイステップ）

### Phase 1: 新規ファイルの配置

1. **新規Pythonファイルを追加**
   ```
   loop-profiler/
   ├── feedback_manager.py      # 新規
   ├── feature_extractor.py     # 新規
   ├── ml_predictor.py          # 新規
   └── main.py                  # 既存（後で修正）
   ```

2. **ファイルのコピー**
   - `feedback_manager.py` をプロジェクトルートに配置
   - `feature_extractor.py` をプロジェクトルートに配置
   - `ml_predictor.py` をプロジェクトルートに配置

---

### Phase 2: main.py の修正

**重要:** 既存のmain.pyをバックアップしてから作業してください

```bash
cp main.py main.py.backup
```

#### 修正箇所1: インポート追加（ファイル冒頭）

```python
# 既存のインポート後に追加
try:
    from feedback_manager import FeedbackManager
    from feature_extractor import FeatureExtractor
    from ml_predictor import MLPredictor
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI機能無効: {e}")
```

#### 修正箇所2: FeatureExtractionWorker クラス追加

`AnalysisWorker` クラスの直後に `FeatureExtractionWorker` クラスを追加
（main_ai_integration.py の該当部分をコピー）

#### 修正箇所3: IntegratedLoopProfiler.__init__() 修正

```python
def __init__(self):
    # ... 既存コード ...
    
    # 【追加】AI機能の初期化（遅延初期化）
    self.feedback_manager = None
    self.feature_extractor = None
    self.ml_predictor = None
    self.feature_worker = None
    
    self.load_bass()
```

#### 修正箇所4: AI初期化メソッド追加

`load_bass()` メソッドの直後に以下を追加:
- `_ensure_ai_initialized()`
- `_train_model_if_needed()`

（main_ai_integration.py の該当部分をコピー）

#### 修正箇所5: parse_and_fill() 修正

candidates.append() の箇所を修正:

```python
# 変更前:
self.candidates.append({"s":s, "e":e, "sc":sc, "f":info.freq, "ch":info.chans})

# 変更後:
self.candidates.append({
    "s": s,
    "e": e,
    "sc": sc,
    "f": info.freq,
    "ch": info.chans,
    "ai_score": None,
    "ai_confidence": None,
    "features": None
})
```

#### 修正箇所6: load_results() 修正

既存のparse_and_fill()呼び出し後にAI分析を追加
（main_ai_integration.py の該当部分をコピー）

#### 修正箇所7: リスト表示メソッド変更

`parse_and_fill()` 内のリスト表示ロジックを `_create_list_item()` メソッドに分離
（main_ai_integration.py の該当部分をコピー）

#### 修正箇所8: 右クリックメニュー追加

`contextMenuEvent()` メソッドを追加
（main_ai_integration.py の該当部分をコピー）

#### 修正箇所9: export_audio() 修正

エクスポート成功後にフィードバック記録を追加
（main_ai_integration.py の該当部分をコピー）

---

### Phase 3: 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

**エラーが出る場合:**

```bash
# 1つずつインストール
pip install scikit-learn
pip install librosa
```

**librosaのインストールに失敗する場合（Windowsの場合）:**

```bash
# Visual C++ Build Toolsが必要な場合があります
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# からインストール後、再試行
```

---

### Phase 4: 動作確認

#### 1. 基本起動テスト

```bash
python main.py
```

**期待される出力:**
```
==================================================
Loop Profiler v2.0 - AI Enhanced
==================================================
✅ AI機能: 有効
==================================================
```

または（librosa未インストール時）:
```
⚠️ AI機能: 無効（基本機能は動作します）
   有効化: pip install librosa scikit-learn
```

#### 2. AI機能テスト

1. **音声ファイルを選択**
   - SELECT AUDIO FILE ボタンをクリック
   - テスト用の.mp3ファイルを選択

2. **解析実行**
   - START ANALYSIS ボタンをクリック
   - "AI ANALYZING... 1/5" のような表示が出ればOK

3. **AI予測確認**
   - リストに "⚡ AI:92.3% ORIG:95.6%" のような表示があればOK

4. **フィードバック記録**
   - リストの項目を右クリック
   - "👍 Mark as Good Loop" を選択
   - "👍 FEEDBACK RECORDED (ID: 1)" と表示されればOK

5. **統計確認**
   - 右クリックメニューから "📊 Show AI Statistics"
   - フィードバック数が表示されればOK

#### 3. モデル学習テスト

10件以上フィードバックを記録すると:
```
✅ モデル学習完了: 10件
🤖 MODEL RETRAINED
```

---

### Phase 5: トラブルシューティング

#### Q1: "ModuleNotFoundError: No module named 'librosa'"

**A1:** librosaをインストール
```bash
pip install librosa
```

#### Q2: AI機能が動作しない（エラーなし）

**A2:** 以下を確認
1. `feedback_manager.py` 等が main.py と同じフォルダにあるか
2. Pythonコンソールで手動インポートできるか確認:
   ```python
   from feedback_manager import FeedbackManager
   ```

#### Q3: 特徴量抽出が遅い（30秒以上かかる）

**A3:** 正常です。初回のみ時間がかかります。2回目以降はキャッシュで高速化されます。

#### Q4: モデルが学習されない

**A4:** 10件以上のフィードバックが必要です。統計を確認:
```python
# 右クリック → "📊 Show AI Statistics"
# Total Feedbacks が 10 以上になっているか確認
```

---

### Phase 6: 最終確認チェックリスト

- [ ] main.py が正常に起動する
- [ ] AI機能が有効と表示される
- [ ] 解析後に AI スコアが表示される
- [ ] 右クリックメニューでフィードバック記録できる
- [ ] 10件フィードバック後にモデルが学習される
- [ ] エクスポート時に自動フィードバックが記録される
- [ ] 既存の基本機能（プレビュー、エクスポート）が動作する

---

## 🎉 完了

すべてのチェックが完了したら、AI機能統合完了です！

次のステップ:
1. 実際の音楽ファイルで使用してフィードバックを蓄積
2. モデルの精度向上を確認
3. 必要に応じてパラメータ調整

---

## 📞 サポート

問題が発生した場合:
1. エラーメッセージをコピー
2. どの手順で問題が発生したかメモ
3. Python/ライブラリのバージョンを確認:
   ```bash
   python --version
   pip list | grep -E "librosa|sklearn|PyQt"
   ```

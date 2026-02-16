# Loop Profiler v2.0 - AI Enhanced

音楽ファイルのループポイントを自動検出し、AI品質予測でベストなループを提案するGUIアプリケーション。

## 🎯 主要機能

### 基本機能
- **自動ループポイント検出**: pymusiclooperを使用したAI解析
- **波形表示**: リアルタイムでループ位置を可視化
- **ループプレビュー**: 検出されたループポイントを即座に試聴
- **柔軟なエクスポート**: イントロ・ループ・アウトロを自由に組み合わせてMP3出力
- **キャッシュ機能**: 一度解析した結果を保存し、次回から即座に読み込み

### 🆕 AI機能 (v2.0)
- **AI品質予測**: 過去のフィードバックから学習し、最適なループポイントを予測
- **音響特徴量解析**: 振幅の滑らかさ、周波数類似度、BPM安定性、音圧バランスを自動分析
- **フィードバック学習**: 👍/👎評価やエクスポート履歴から継続的に学習
- **⚡推奨マーク**: AI予測スコア90%以上のループに⚡⚡マーク、75%以上に⚡マーク表示

## 📋 必要要件

### システム要件
- Windows OS
- Python 3.8以上

### 必須ファイル
- `bass.dll` - BASS audio libraryのDLL（本プログラムと同じフォルダに配置）
  - [BASS公式サイト](https://www.un4seen.com/)からダウンロード可能

### 外部ツール
- `pymusiclooper` - コマンドラインツール（pipでインストール後、システムPATHに追加が必要）

## 🚀 インストール

### 1. リポジトリをクローン

```bash
git clone <repository-url>
cd loop-profiler
```

### 2. 依存パッケージをインストール

#### 基本機能のみ（AI機能なし）
```bash
pip install PyQt6 pydub pymusiclooper
```

#### AI機能を含む（推奨）
```bash
pip install -r requirements.txt
```

### 3. bass.dllを配置

`bass.dll`を入手し、`main.py`と同じフォルダに配置

### 4. pymusiclooperの確認

```bash
pymusiclooper --version
```

システムPATHに含まれていない場合は追加してください。

## 📖 使い方

### 基本的な使い方

1. **アプリケーションを起動**
   ```bash
   python main.py
   ```

2. **SELECT AUDIO FILE**ボタンでMP3/WAV/OGGファイルを選択

3. **START ANALYSIS**ボタンで解析開始（初回のみ。2回目以降はキャッシュから自動読み込み）

4. リストから好みのループポイントをダブルクリックでプレビュー
   - ★マークは95%以上の高精度ループポイント
   - ⚡マークはAI推奨ループポイント（75%以上）
   - ⚡⚡マークはAI超推奨ループポイント（90%以上）

5. **EXPORT SETTINGS**でエクスポート設定を調整
   - **Intro**: ループ開始前の部分を含めるか
   - **Loop Section**: ループ区間を含めるか
   - **Outro**: ループ終了後の部分を含めるか
   - **Loop Repeat Count**: ループを何回繰り返すか（1〜100回）

6. **EXPORT TO MP3**ボタンで320kbps MP3として保存

### AI機能の使い方

#### フィードバックを記録
1. ループ候補を**右クリック**
2. **👍 Mark as Good Loop** または **👎 Mark as Bad Loop** を選択
3. 自動的にフィードバックが記録され、モデルが再学習されます

#### AI統計を確認
1. リストを**右クリック**
2. **📊 Show AI Statistics** を選択
3. フィードバック数、モデル性能を確認

#### 自動学習
- エクスポート成功時に自動的に👍として記録
- 10件以上のフィードバックでモデルが学習開始
- フィードバックを追加するたびにモデルが自動的に再学習

## 🗂️ プロジェクト構成

```
loop-profiler/
├── main.py                      # メインプログラム
├── feedback_manager.py          # フィードバック管理
├── feature_extractor.py         # 特徴量抽出
├── ml_predictor.py              # ML予測
├── bass.dll                     # BASSライブラリ（要配置）
├── requirements.txt             # Python依存パッケージ
├── README.md                    # このファイル
├── INTEGRATION_GUIDE.md         # 統合手順書
└── LooperOutput/                # 解析結果・学習データ保存先（自動作成）
    ├── *.txt                    # 解析結果キャッシュ
    ├── feedback.json            # フィードバックDB
    ├── loop_model.pkl           # 学習済みモデル
    └── features_cache/          # 特徴量キャッシュ
```

## 🎨 UI説明

### 波形表示
- **青い波形**: 曲全体の波形
- **緑の点線（縦線）**: ループ開始位置とループ終了位置
- **緑の半透明エリア**: ループ区間
- **赤い縦線**: 現在の再生位置

### リスト表示

#### AI機能有効時
```
⚡⚡ AI:92.3%  ORIG:95.6%  |  Start: 0:06  End: 0:30  Loop: 24.0s
```
- **⚡⚡**: AI超推奨（90%以上）
- **⚡**: AI推奨（75%以上）
- **AI**: AI予測スコア
- **ORIG**: Pymusiclooper元スコア

#### AI機能無効時
```
★ SCORE: 95.60%  |  Start: 0:06  End: 0:30  Loop: 24.0s
```
- **★**: 高精度ループ（95%以上）

## 🔧 トラブルシューティング

### `bass.dll not found`エラー
- `bass.dll`が`main.py`と同じフォルダにあるか確認してください

### `pymusiclooper`が見つからない
- コマンドプロンプトで`pymusiclooper --version`が動作するか確認
- 動作しない場合はPATHの設定を確認してください

### AI機能が動作しない
```
⚠️ AI機能の依存ライブラリが見つかりません
```
と表示される場合:
```bash
pip install librosa scikit-learn
```

### 特徴量抽出が遅い
- 初回のみ2〜3秒かかります
- 2回目以降はキャッシュで高速化されます

### モデルが学習されない
- 最低10件のフィードバックが必要です
- 右クリック → 📊 Show AI Statistics で確認

### 波形表示がおかしい
- ファイルを選択し直すと正常に表示されることがあります
- 対応フォーマット（MP3/WAV/OGG）であることを確認してください

### エクスポート時のエラー
- 少なくとも1つのセクション（イントロ/ループ/アウトロ）が選択されているか確認
- ディスク容量が十分にあるか確認
- 保存先のパスに日本語や特殊文字が含まれていないか確認

## 📊 データ管理

### キャッシュをクリアしたい
`LooperOutput`フォルダ内の以下を削除:
- `.txt`ファイル: 解析結果キャッシュ
- `features_cache/`: 特徴量キャッシュ

### フィードバックをリセットしたい
`LooperOutput/feedback.json`を削除（バックアップ推奨）

### 学習済みモデルをリセットしたい
`LooperOutput/loop_model.pkl`を削除

## 📝 ライセンス

このプロジェクトは個人・商用利用可能です。ただし以下の外部ライブラリのライセンスに従ってください：

- BASS audio library: [ライセンス条項](https://www.un4seen.com/)
- pymusiclooper: MIT License
- scikit-learn: BSD License
- librosa: ISC License

## 🙏 クレジット

- GUI: PyQt6
- 音声処理: BASS audio library, pydub
- ループ検出エンジン: pymusiclooper
- AI/ML: scikit-learn, librosa

## 📜 変更履歴

### v2.0.0 (2026-02-16)
- ✨ AI品質予測機能を追加
- ✨ 音響特徴量解析（振幅、周波数、BPM、音圧）
- ✨ フィードバック学習機能
- ✨ ⚡推奨マーク表示
- 🔧 UI改善（右クリックメニュー、AI統計表示）

### v1.0.0
- 初回リリース
- ループポイント自動検出機能
- 波形表示機能
- エクスポート機能（イントロ・ループ・アウトロの組み合わせ）
- キャッシュ機能

## 🤝 コントリビューション

バグ報告や機能要望はIssueでお願いします。

## 📞 サポート

問題が発生した場合は、以下の情報を含めてIssueを作成してください：
- エラーメッセージ
- 操作手順
- Python/ライブラリのバージョン
```bash
python --version
pip list | grep -E "PyQt6|pydub|librosa|sklearn"
```

import os
import sys
import ctypes
import subprocess
import glob
import shutil

try:
    from mutagen.oggvorbis import OggVorbis
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog,
    QLabel, QMessageBox, QProgressBar, QFrame, QListWidgetItem,
    QSpinBox, QGroupBox, QCheckBox, QMenu
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QLinearGradient, QFont
from pydub import AudioSegment
from pathlib import Path

# ============================================================
# AI機能のインポート
# ============================================================
try:
    from feedback_manager import FeedbackManager
    from feature_extractor import FeatureExtractor
    from ml_predictor import MLPredictor
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI機能の依存ライブラリが見つかりません: {e}")
    print("   インストール: pip install librosa scikit-learn")
    print("   → AI品質予測は無効化されますが、基本機能は動作します")

# ============================================================
# BASS Constants
# ============================================================
BASS_POS_BYTE = 0
BASS_SYNC_POS = 0
BASS_SYNC_MIXTIME = 0x40000000
BASS_UNICODE = 0x80000000
BASS_STREAM_DECODE = 0x200000
BASS_SAMPLE_FLOAT = 256
BASS_STREAM_PRESCAN = 0x20000

SYNCFUNC_TYPE = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p
)

class BASS_CHANNELINFO(ctypes.Structure):
    _fields_ = [
        ("freq", ctypes.c_uint32), ("chans", ctypes.c_uint32),
        ("flags", ctypes.c_uint32), ("ctype", ctypes.c_uint32),
        ("origres", ctypes.c_uint32), ("plugin", ctypes.c_void_p),
        ("sample", ctypes.c_void_p), ("filename", ctypes.c_char_p),
    ]

# ============================================================
# Analysis Worker (pymusiclooperを非同期実行)
# ============================================================
class AnalysisWorker(QThread):
    finished = pyqtSignal(bool, str)
    status = pyqtSignal(str)

    def __init__(self, audio_path, output_dir, brute_force=False):
        super().__init__()
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.brute_force = brute_force

    def run(self):
        mode_text = "BRUTE-FORCE MODE" if self.brute_force else "AI ENGINE"
        self.status.emit(f"RUNNING {mode_text}...")
        try:
            cmd = [
                "pymusiclooper", "export-points", "--path", self.audio_path,
                "--alt-export-top", "5", "--export-to", "txt", "--output-dir", self.output_dir
            ]
            
            # Brute-forceモードの場合は --brute-force オプションを追加
            if self.brute_force:
                cmd.append("--brute-force")
            
            subprocess.run(cmd, check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.finished.emit(True, "SUCCESS")
        except Exception as e:
            self.finished.emit(False, str(e))

# ============================================================
# Feature Extraction Worker (特徴量抽出を非同期実行)
# ============================================================
class FeatureExtractionWorker(QThread):
    """特徴量抽出をバックグラウンドで実行"""
    finished = pyqtSignal(list)  # [(candidate_idx, features), ...]
    progress = pyqtSignal(int, int)  # (current, total)
    
    def __init__(self, audio_path, candidates, sample_rate, feature_extractor):
        super().__init__()
        self.audio_path = audio_path
        self.candidates = candidates
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor
    
    def run(self):
        results = []
        
        for i, c in enumerate(self.candidates):
            try:
                features = self.feature_extractor.extract(
                    self.audio_path,
                    c["s"],
                    c["e"],
                    self.sample_rate,
                    use_cache=True
                )
                results.append((i, features))
            except Exception as e:
                print(f"⚠️ 特徴量抽出エラー (候補{i}): {e}")
                # エラー時はデフォルト値
                results.append((i, {
                    "amplitude_smoothness": 0.5,
                    "spectral_similarity": 0.5,
                    "tempo_consistency": 0.5,
                    "loudness_matching": 0.5
                }))
            
            self.progress.emit(i + 1, len(self.candidates))
        
        self.finished.emit(results)

# ============================================================
# Waveform Processing Worker
# ============================================================
class WaveformLoader(QThread):
    finished = pyqtSignal(list, int)

    def __init__(self, bass_dll, file_path):
        super().__init__()
        self.bass = bass_dll
        self.file_path = file_path

    def run(self):
        flags = BASS_UNICODE | BASS_STREAM_DECODE | BASS_SAMPLE_FLOAT | BASS_STREAM_PRESCAN
        handle = self.bass.BASS_StreamCreateFile(False, ctypes.c_wchar_p(self.file_path), 0, 0, flags)
        if not handle: return

        total_bytes = self.bass.BASS_ChannelGetLength(handle, BASS_POS_BYTE)
        if total_bytes <= 0:
            self.bass.BASS_StreamFree(handle)
            return

        num_points = 1500
        bytes_per_point = total_bytes // num_points
        peaks = []
        buffer = (ctypes.c_float * 1024)()
        
        for i in range(num_points):
            self.bass.BASS_ChannelSetPosition(handle, ctypes.c_uint64(i * bytes_per_point), BASS_POS_BYTE)
            read_len = self.bass.BASS_ChannelGetData(handle, buffer, 1024 | BASS_POS_BYTE)
            if read_len <= 0: break
            
            count = read_len // 4
            m = max((abs(buffer[j]) for j in range(count)), default=0.0)
            peaks.append(m)

        self.bass.BASS_StreamFree(handle)
        self.finished.emit(peaks, total_bytes)

# ============================================================
# Waveform Widget
# ============================================================
class WaveformWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.peaks, self.total_bytes = [], 1
        self.play_pos, self.loop_start, self.loop_end = 0, 0, 0

    def set_data(self, peaks, total_bytes):
        self.peaks, self.total_bytes = peaks, max(1, total_bytes)
        self.update()

    def set_position(self, pos, l_start, l_end):
        self.play_pos, self.loop_start, self.loop_end = pos, l_start, l_end
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h, mid_h = self.width(), self.height(), self.height() / 2

        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0, QColor(15, 15, 15)); bg_grad.setColorAt(1, QColor(25, 25, 25))
        p.fillRect(0, 0, w, h, bg_grad)

        if self.peaks:
            wave_grad = QLinearGradient(0, 0, 0, h)
            wave_grad.setColorAt(0, QColor(0, 255, 255)); wave_grad.setColorAt(0.5, QColor(0, 100, 255)); wave_grad.setColorAt(1, QColor(0, 255, 255))
            p.setPen(QPen(wave_grad, 1))
            step_x = w / len(self.peaks)
            for i, val in enumerate(self.peaks):
                x = int(i * step_x)
                amp = int(val * (h * 0.8) / 2)
                p.drawLine(x, int(mid_h - amp), x, int(mid_h + amp))

        if self.loop_end > 0:
            sx, ex = (self.loop_start / self.total_bytes) * w, (self.loop_end / self.total_bytes) * w
            p.fillRect(QRect(int(sx), 0, int(ex - sx), h), QColor(0, 255, 255, 35))
            p.setPen(QPen(QColor(0, 255, 255, 150), 1, Qt.PenStyle.DashLine))
            p.drawLine(int(sx), 0, int(sx), h); p.drawLine(int(ex), 0, int(ex), h)

        cx = (self.play_pos / self.total_bytes) * w
        p.setPen(QPen(QColor(255, 0, 100), 2))
        p.drawLine(int(cx), 0, int(cx), h)

# ============================================================
# Main Application
# ============================================================
class IntegratedLoopProfiler(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loop Profiler v2.0 - Powered by Pymusiclooper & BASS & AI")
        self.resize(750, 900)
        self.apply_styles()
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.script_dir, "LooperOutput")
        os.makedirs(self.output_dir, exist_ok=True)

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        self.label = QLabel("READY")
        self.label.setObjectName("titleLabel")
        layout.addWidget(self.label)

        self.btn_select = QPushButton("SELECT AUDIO FILE")
        self.btn_select.clicked.connect(self.select_audio)
        layout.addWidget(self.btn_select)

        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)

        status_box = QHBoxLayout()
        self.status_label = QLabel("SYSTEM IDLE")
        self.status_label.setStyleSheet("color: #00D2FF; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        status_box.addWidget(self.status_label)
        status_box.addStretch()
        layout.addLayout(status_box)

        # Brute-forceモードチェックボックス
        brute_force_layout = QHBoxLayout()
        self.check_brute_force = QCheckBox("🔍 Brute-Force Mode (slower, more accurate)")
        self.check_brute_force.setStyleSheet("""
            QCheckBox { 
                color: #888; 
                font-size: 10px;
                spacing: 8px;
            }
            QCheckBox::indicator { 
                width: 14px; 
                height: 14px; 
                border-radius: 3px; 
                border: 2px solid #555;
                background-color: #1a1a1a;
            }
            QCheckBox::indicator:checked { 
                background-color: #ff9900; 
                border: 2px solid #ff9900; 
            }
        """)
        self.check_brute_force.setToolTip(
            "Brute-Force Mode: より徹底的な解析を行います。\n"
            "処理時間は長くなりますが、より多くのループポイント候補を検出できます。"
        )
        brute_force_layout.addWidget(self.check_brute_force)
        brute_force_layout.addStretch()
        layout.addLayout(brute_force_layout)

        self.btn_analyze = QPushButton("START ANALYSIS")
        self.btn_analyze.setObjectName("analyzeBtn")
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_analyze.setEnabled(False)
        layout.addWidget(self.btn_analyze)

        self.progress = QProgressBar()
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self.on_candidate_selected)
        layout.addWidget(self.list)

        # 編集パネルの追加
        edit_group = self.create_edit_panel()
        layout.addWidget(edit_group)

        self.btn_stop = QPushButton("STOP PREVIEW")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self.stop_audio)
        layout.addWidget(self.btn_stop)

        self.setLayout(layout)

        self.audio_path, self.handle, self.loop_sync = None, 0, 0
        self._loop_callback_func, self.current_loop_bytes = None, (0, 0)
        self.candidates = []
        self.audio_duration_ms = 0
        
        # AI機能の初期化（遅延初期化）
        self.feedback_manager = None
        self.feature_extractor = None
        self.ml_predictor = None
        self.feature_worker = None
        
        self.timer = QTimer(); self.timer.timeout.connect(self.update_ui); self.timer.start(30)
        self.load_bass()

    def create_edit_panel(self):
        group = QGroupBox("EXPORT SETTINGS")
        group.setStyleSheet("""
            QGroupBox { 
                color: #00D2FF; 
                font-weight: bold; 
                border: 1px solid #333; 
                border-radius: 4px; 
                margin-top: 12px; 
                padding-top: 16px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)
        
        layout = QVBoxLayout()
        
        # セクション選択
        section_label = QLabel("SECTIONS TO INCLUDE:")
        section_label.setStyleSheet("color: #00D2FF; font-size: 11px; font-weight: bold; margin-top: 6px;")
        layout.addWidget(section_label)
        
        section_layout = QHBoxLayout()
        self.check_intro = QCheckBox("Intro (before loop)")
        self.check_loop = QCheckBox("Loop Section")
        self.check_outro = QCheckBox("Outro (after loop)")
        
        self.check_intro.setChecked(True)
        self.check_loop.setChecked(True)
        self.check_outro.setChecked(True)
        
        section_layout.addWidget(self.check_intro)
        section_layout.addWidget(self.check_loop)
        section_layout.addWidget(self.check_outro)
        section_layout.addStretch()
        layout.addLayout(section_layout)
        
        # ループ回数設定
        loop_count_layout = QHBoxLayout()
        loop_count_layout.addWidget(QLabel("Loop Repeat Count:"))
        self.spin_loop_count = QSpinBox()
        self.spin_loop_count.setRange(1, 100)
        self.spin_loop_count.setValue(3)
        self.spin_loop_count.setFixedWidth(80)
        loop_count_layout.addWidget(self.spin_loop_count)
        loop_count_layout.addStretch()
        layout.addLayout(loop_count_layout)
        
        # プレビュー情報
        self.preview_label = QLabel("")
        self.preview_label.setStyleSheet("color: #888; font-size: 10px; margin: 8px 0;")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)
        
        # エクスポートボタン
        self.btn_export = QPushButton("EXPORT TO MP3")
        self.btn_export.setObjectName("exportBtn")
        self.btn_export.clicked.connect(self.export_audio)
        self.btn_export.setEnabled(False)
        layout.addWidget(self.btn_export)

        # OGGエクスポートボタン（入力がOGGのときのみ有効）
        self.btn_export_ogg = QPushButton("EXPORT TO OGG  (with LOOPSTART / LOOPLENGTH tags)")
        self.btn_export_ogg.setObjectName("exportOggBtn")
        self.btn_export_ogg.clicked.connect(self.export_ogg_with_tags)
        self.btn_export_ogg.setEnabled(False)
        layout.addWidget(self.btn_export_ogg)
        
        # チェックボックスの変更を監視
        self.check_intro.stateChanged.connect(self.update_preview_info)
        self.check_loop.stateChanged.connect(self.update_preview_info)
        self.check_outro.stateChanged.connect(self.update_preview_info)
        self.spin_loop_count.valueChanged.connect(self.update_preview_info)
        
        group.setLayout(layout)
        return group

    def update_preview_info(self):
        """エクスポート設定のプレビュー情報を更新"""
        if not self.candidates or self.list.currentRow() < 0:
            self.preview_label.setText("")
            return
        
        row = self.list.currentRow()
        c = self.candidates[row]
        
        # サンプル→ミリ秒変換
        start_ms = (c["s"] * 1000) // c["f"]
        end_ms = (c["e"] * 1000) // c["f"]
        loop_section_ms = end_ms - start_ms
        
        # 各セクションの長さを計算
        intro_duration = start_ms if self.check_intro.isChecked() else 0
        loop_duration = loop_section_ms * self.spin_loop_count.value() if self.check_loop.isChecked() else 0
        outro_duration = 0
        
        if self.check_outro.isChecked() and self.audio_duration_ms > 0:
            outro_duration = self.audio_duration_ms - end_ms
        
        total_duration = intro_duration + loop_duration + outro_duration
        total_sec = total_duration / 1000
        
        # 構成を表示
        parts = []
        if self.check_intro.isChecked():
            parts.append(f"Intro ({intro_duration/1000:.1f}s)")
        if self.check_loop.isChecked():
            parts.append(f"Loop×{self.spin_loop_count.value()} ({loop_duration/1000:.1f}s)")
        if self.check_outro.isChecked():
            parts.append(f"Outro ({outro_duration/1000:.1f}s)")
        
        if parts:
            composition = " + ".join(parts)
            self.preview_label.setText(f"Composition: {composition}\nTotal Duration: {total_sec:.1f}s ({int(total_sec//60)}:{int(total_sec%60):02d})")
        else:
            self.preview_label.setText("No sections selected")

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { background-color: #101010; color: #888; font-family: 'Segoe UI', 'Arial'; }
            QLabel#titleLabel { color: #fff; font-size: 14px; font-weight: bold; }
            QPushButton { 
                background-color: #222; border-radius: 4px; padding: 12px; color: #bbb; font-weight: bold; border: 1px solid #333;
            }
            QPushButton:hover { background-color: #282828; border: 1px solid #00D2FF; color: #fff; }
            QPushButton:disabled { color: #444; background-color: #151515; border: 1px solid #222; }
            QPushButton#analyzeBtn { background-color: #002a35; color: #00D2FF; border: 1px solid #00D2FF; }
            QPushButton#stopBtn { color: #ff3366; background-color: #201015; border: 1px solid #442222; }
            QPushButton#exportBtn { 
                background-color: #1a3a1a; 
                color: #00ff88; 
                border: 1px solid #00ff88; 
                font-size: 12px;
            }
            QPushButton#exportBtn:hover { 
                background-color: #244a24; 
            }
            QPushButton#exportOggBtn { 
                background-color: #2a2010; 
                color: #ff9900; 
                border: 1px solid #ff9900; 
                font-size: 12px;
            }
            QPushButton#exportOggBtn:hover { 
                background-color: #3a2a10; 
            }
            QListWidget { background-color: #080808; border: 1px solid #222; border-radius: 4px; outline: none; }
            QListWidget::item { background-color: #151515; margin: 4px; padding: 14px; border-radius: 4px; border-left: 5px solid #222; }
            QListWidget::item:selected { border-left: 5px solid #ff3366; background-color: #202020; color: #fff; }
            QProgressBar { background-color: #080808; border: none; }
            QProgressBar::chunk { background-color: #00D2FF; }
            QSpinBox { 
                background-color: #1a1a1a; 
                border: 1px solid #333; 
                padding: 4px; 
                color: #bbb; 
                border-radius: 2px;
            }
            QCheckBox { 
                color: #bbb; 
                spacing: 8px;
            }
            QCheckBox::indicator { 
                width: 16px; 
                height: 16px; 
                border-radius: 3px; 
                border: 2px solid #555;
                background-color: #1a1a1a;
            }
            QCheckBox::indicator:checked { 
                background-color: #00D2FF; 
                border: 2px solid #00D2FF; 
            }
        """)

    def load_bass(self):
        dll_path = os.path.join(self.script_dir, "bass.dll")
        if not os.path.exists(dll_path):
            QMessageBox.critical(self, "Error", "bass.dll not found")
            sys.exit(1)
        self.bass = ctypes.WinDLL(dll_path)
        self.bass.BASS_Init(-1, 44100, 0, 0, 0)

    # ============================================================
    # AI機能の初期化メソッド
    # ============================================================
    def _ensure_ai_initialized(self):
        """AI機能を初回使用時に初期化"""
        if not AI_AVAILABLE:
            return False
        
        if self.feedback_manager is not None:
            return True  # 既に初期化済み
        
        try:
            output_dir = Path(self.output_dir)
            
            self.feedback_manager = FeedbackManager(
                str(output_dir / "feedback.json")
            )
            self.feature_extractor = FeatureExtractor(
                str(output_dir / "features_cache")
            )
            self.ml_predictor = MLPredictor(
                str(output_dir / "loop_model.pkl")
            )
            
            # 既存フィードバックでモデルを学習
            self._train_model_if_needed()
            
            print("✅ AI機能初期化完了")
            return True
        
        except Exception as e:
            print(f"❌ AI機能初期化エラー: {e}")
            self.feedback_manager = None
            return False
    
    def _train_model_if_needed(self):
        """既存フィードバックでモデルを学習"""
        if self.feedback_manager is None or self.ml_predictor is None:
            return
        
        X, y = self.feedback_manager.get_training_data()
        
        if len(X) >= MLPredictor.MIN_TRAINING_SAMPLES:
            success = self.ml_predictor.train(X, y)
            
            if success:
                # モデル性能を記録
                eval_result = self.ml_predictor.evaluate(X, y)
                self.feedback_manager.update_model_performance(
                    accuracy=eval_result.get("accuracy", 0),
                    precision=eval_result.get("precision", 0),
                    recall=eval_result.get("recall", 0),
                    n_samples=len(X)
                )

    def select_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio", "", "Audio (*.mp3 *.wav *.ogg)")
        if path:
            self.audio_path = os.path.normpath(path)
            self.label.setText(os.path.basename(path).upper())
            self.btn_analyze.setEnabled(True)
            self.status_label.setText("FILE READY")
            # OGGエクスポートボタンはOGG入力のときのみ有効化
            is_ogg = path.lower().endswith(".ogg")
            self.btn_export_ogg.setEnabled(False)  # 候補選択後に有効化するためリセット
            self.btn_export_ogg.setToolTip(
                "OGGファイルを選択したときのみ使用できます" if not is_ogg else
                "元のOGGファイルにLOOPSTART/LOOPLENGTHタグを付与して保存します"
            )
            self.load_audio_stream()

    def load_audio_stream(self):
        if self.handle: self.bass.BASS_StreamFree(self.handle)
        flags = BASS_UNICODE | BASS_STREAM_PRESCAN
        self.handle = self.bass.BASS_StreamCreateFile(False, ctypes.c_wchar_p(self.audio_path), 0, 0, flags)
        if self.handle:
            # 曲の長さをキャッシュ
            try:
                audio = AudioSegment.from_file(self.audio_path)
                self.audio_duration_ms = len(audio)
            except:
                self.audio_duration_ms = 0
            
            # 再生用ストリームのtotal_bytesを取得
            playback_total_bytes = self.bass.BASS_ChannelGetLength(self.handle, BASS_POS_BYTE)
            
            self.wave_worker = WaveformLoader(self.bass, self.audio_path)
            # 波形読み込み完了後、total_bytesを再生用ストリームのものに上書き
            self.wave_worker.finished.connect(lambda peaks, _: self.waveform.set_data(peaks, playback_total_bytes))
            self.wave_worker.start()
            # 曲を読み込んだ瞬間に解析結果(キャッシュ)があるかチェック
            self.run_analysis(auto_check=True)

    def run_analysis(self, auto_check=False):
        if not self.audio_path: return
        
        # Brute-forceモードのチェック状態を取得
        brute_force_mode = self.check_brute_force.isChecked()
        
        # 1. 既存の解析結果(キャッシュ)があるかチェック
        base = os.path.basename(self.audio_path)
        
        # Brute-forceモードの場合は別のキャッシュファイル名を使用
        if brute_force_mode:
            cache_pattern = os.path.join(self.output_dir, f"{base}*_brute.txt")
        else:
            cache_pattern = os.path.join(self.output_dir, f"{base}*.txt")
            # Brute-forceキャッシュは除外
            cache_pattern_exclude = os.path.join(self.output_dir, f"{base}*_brute.txt")
        
        existing_files = glob.glob(cache_pattern)
        
        # 通常モードの場合、brute-forceキャッシュを除外
        if not brute_force_mode:
            brute_files = glob.glob(cache_pattern_exclude)
            existing_files = [f for f in existing_files if f not in brute_files]
        
        if existing_files:
            mode_text = "BRUTE-FORCE " if brute_force_mode else ""
            self.status_label.setText(f"{mode_text}CACHED DATA FOUND. LOADING...")
            self.load_results()
            self.status_label.setText(f"{mode_text}CACHE LOADED - READY")
            return
        
        # 自動チェックモードの時は、キャッシュがない場合は何もしない(ボタン押し待ち)
        if auto_check:
            self.status_label.setText("NO CACHE. READY FOR ANALYSIS.")
            return

        # 2. キャッシュがない場合のみ Pymusiclooper を別スレッドで実行
        self.btn_analyze.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.check_brute_force.setEnabled(False)  # 解析中は変更不可
        self.progress.setRange(0, 0) 
        
        self.analysis_worker = AnalysisWorker(self.audio_path, self.output_dir, brute_force=brute_force_mode)
        self.analysis_worker.status.connect(lambda s: self.status_label.setText(s))
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.start()

    def on_analysis_finished(self, success, message):
        self.progress.setRange(0, 100); self.progress.setValue(100)
        self.btn_analyze.setEnabled(True); self.btn_select.setEnabled(True)
        self.check_brute_force.setEnabled(True)  # 解析完了後に再度有効化
        if success:
            mode_text = "BRUTE-FORCE " if self.check_brute_force.isChecked() else ""
            self.status_label.setText(f"{mode_text}ANALYSIS COMPLETE")
            self.load_results()
        else:
            self.status_label.setText("ENGINE ERROR")
            QMessageBox.warning(self, "Analysis Error", message)

    def load_results(self):
        """解析結果を読み込み + AI分析を開始"""
        base = os.path.basename(self.audio_path)
        files = glob.glob(os.path.join(self.output_dir, f"{base}*.txt"))
        if files:
            target = max(files, key=os.path.getmtime)
            self.parse_and_fill(target)
            
            # AI分析をバックグラウンドで開始
            if AI_AVAILABLE and self._ensure_ai_initialized():
                info = BASS_CHANNELINFO()
                self.bass.BASS_ChannelGetInfo(self.handle, ctypes.byref(info))
                
                self.feature_worker = FeatureExtractionWorker(
                    self.audio_path,
                    self.candidates,
                    info.freq,
                    self.feature_extractor
                )
                self.feature_worker.progress.connect(
                    lambda cur, total: self.status_label.setText(
                        f"AI ANALYZING... {cur}/{total}"
                    )
                )
                self.feature_worker.finished.connect(self._on_features_extracted)
                self.feature_worker.start()

    def _on_features_extracted(self, results):
        """特徴量抽出完了時の処理"""
        for idx, features in results:
            self.candidates[idx]["features"] = features
            
            # AI予測実行
            if self.ml_predictor and self.ml_predictor.is_trained:
                ai_score, confidence = self._get_ai_prediction(self.candidates[idx])
                self.candidates[idx]["ai_score"] = ai_score
                self.candidates[idx]["ai_confidence"] = confidence
            else:
                self.candidates[idx]["ai_score"] = None
                self.candidates[idx]["ai_confidence"] = None
            
            # リスト表示更新
            self.list.takeItem(idx)
            new_item = self._create_list_item(idx)
            self.list.insertItem(idx, new_item)
        
        self.status_label.setText("AI ANALYSIS COMPLETE")
    
    def _get_ai_prediction(self, candidate: dict):
        """候補のAI予測スコアを取得"""
        if self.ml_predictor is None or not self.ml_predictor.is_trained:
            return None, None
        
        features = candidate.get("features")
        if features is None:
            return None, None
        
        feature_vector = [
            candidate["sc"] / 100.0,  # pymusiclooper (0-1)
            features.get("amplitude_smoothness", 0.0),
            features.get("spectral_similarity", 0.0),
            features.get("tempo_consistency", 0.0),
            features.get("loudness_matching", 0.0)
        ]
        
        return self.ml_predictor.predict(feature_vector)

    def parse_and_fill(self, path):
        self.list.clear(); self.candidates = []
        info = BASS_CHANNELINFO()
        self.bass.BASS_ChannelGetInfo(self.handle, ctypes.byref(info))
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p)>=5:
                    try:
                        s, e, sc = int(p[0]), int(p[1]), float(p[4])*100
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
                        
                        # 初期表示(AIスコアなし)
                        item = self._create_list_item(len(self.candidates) - 1)
                        self.list.addItem(item)
                    except: continue

    def _create_list_item(self, idx):
        """リストアイテムを生成(AIスコア対応)"""
        c = self.candidates[idx]
        
        # AI スコア(あれば)
        if c.get("ai_score") is not None:
            ai_score = c["ai_score"]
            if ai_score >= 90:
                ai_tag = " ⚡⚡"  # 超推奨
                color = QColor("#00ff88")
            elif ai_score >= 75:
                ai_tag = " ⚡"   # 推奨
                color = QColor("#00D2FF")
            else:
                ai_tag = ""
                color = QColor("#888")
            
            score_text = f"{ai_tag} AI:{ai_score:>5.1f}%  ORIG:{c['sc']:>5.1f}%"
        else:
            # AI未計算(従来通り)
            orig_tag = " ★" if c["sc"] >= 95 else "  "
            score_text = f"{orig_tag} SCORE: {c['sc']:>6.2f}%"
            color = QColor("#00D2FF") if c["sc"] >= 95 else QColor("#888")
        
        # 時刻計算
        start_sec = c["s"] / c["f"]
        end_sec = c["e"] / c["f"]
        loop_len_sec = (c["e"] - c["s"]) / c["f"]
        
        item_text = (
            f"{score_text}  |  "
            f"Start: {int(start_sec//60)}:{int(start_sec%60):02d}.{int((start_sec%1)*10)}  "
            f"End: {int(end_sec//60)}:{int(end_sec%60):02d}.{int((end_sec%1)*10)}  "
            f"Loop: {loop_len_sec:.1f}s  |  "
            f"{c['s']:,} → {c['e']:,}  ({c['e'] - c['s']:,})"
        )
        
        item = QListWidgetItem(item_text)
        item.setFont(QFont("Consolas", 9))
        item.setForeground(color)
        return item

    def on_candidate_selected(self, item):
        row = self.list.currentRow()
        if row < 0: return
        c = self.candidates[row]
        s_bytes, e_bytes = c["s"] * 2 * c["ch"], c["e"] * 2 * c["ch"]
        
        if self.loop_sync: self.bass.BASS_ChannelRemoveSync(self.handle, self.loop_sync)
        
        def cb(h, ch, d, u): self.bass.BASS_ChannelSetPosition(ch, ctypes.c_uint64(s_bytes), BASS_POS_BYTE)
        self._loop_callback_func = SYNCFUNC_TYPE(cb)
        self.loop_sync = self.bass.BASS_ChannelSetSync(self.handle, BASS_SYNC_POS|BASS_SYNC_MIXTIME, ctypes.c_uint64(e_bytes), self._loop_callback_func, None)
        
        preroll = int(c["f"] * c["ch"] * 2 * 5.0)
        self.bass.BASS_ChannelSetPosition(self.handle, ctypes.c_uint64(max(0, e_bytes - preroll)), BASS_POS_BYTE)
        self.bass.BASS_ChannelPlay(self.handle, False)
        self.current_loop_bytes = (s_bytes, e_bytes)
        self.status_label.setText(f"LOOP PREVIEW: {c['sc']:.1f}% ACCURACY")
        self.btn_export.setEnabled(True)
        # OGGエクスポートは入力がOGGかつmutagenが利用可能なときのみ有効
        if self.audio_path and self.audio_path.lower().endswith(".ogg") and MUTAGEN_AVAILABLE:
            self.btn_export_ogg.setEnabled(True)
        self.update_preview_info()

    # ============================================================
    # 右クリックメニュー(フィードバック記録)
    # ============================================================
    def contextMenuEvent(self, event):
        """右クリックメニュー"""
        if self.list.underMouse() and AI_AVAILABLE:
            menu = QMenu(self)
            
            good_action = menu.addAction("👍 Mark as Good Loop")
            bad_action = menu.addAction("👎 Mark as Bad Loop")
            menu.addSeparator()
            stats_action = menu.addAction("📊 Show AI Statistics")
            
            action = menu.exec(event.globalPos())
            
            if action == good_action:
                self._record_feedback(rating=1, source="thumbs_up")
            elif action == bad_action:
                self._record_feedback(rating=0, source="thumbs_down")
            elif action == stats_action:
                self._show_ai_statistics()
    
    def _record_feedback(self, rating: int, source: str = "manual"):
        """フィードバックを記録してモデル再学習"""
        row = self.list.currentRow()
        if row < 0:
            return
        
        if not self._ensure_ai_initialized():
            QMessageBox.information(self, "AI Not Available",
                "AI機能を使用するには以下をインストールしてください:\n"
                "pip install librosa scikit-learn")
            return
        
        c = self.candidates[row]
        
        # フィードバック追加
        try:
            feedback_id = self.feedback_manager.add_feedback(
                audio_path=self.audio_path,
                loop_start=c["s"],
                loop_end=c["e"],
                features=c.get("features", {}),
                rating=rating,
                pymusiclooper_score=c["sc"] / 100.0,
                source=source
            )
            
            # モデル再学習(非同期)
            QTimer.singleShot(100, self._retrain_model_async)
            
            # 通知
            emoji = "👍" if rating == 1 else "👎"
            self.status_label.setText(f"{emoji} FEEDBACK RECORDED (ID: {feedback_id})")
        
        except Exception as e:
            QMessageBox.warning(self, "Error", f"フィードバック記録エラー: {e}")
    
    def _retrain_model_async(self):
        """モデルを非同期で再学習"""
        if self.feedback_manager is None or self.ml_predictor is None:
            return
        
        X, y = self.feedback_manager.get_training_data()
        
        if len(X) >= MLPredictor.MIN_TRAINING_SAMPLES:
            self.ml_predictor.train(X, y)
            self.status_label.setText("🤖 MODEL RETRAINED")
    
    def _show_ai_statistics(self):
        """AI統計情報を表示"""
        if self.feedback_manager is None:
            return
        
        stats = self.feedback_manager.get_statistics()
        
        msg = f"""AI Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━
Total Feedbacks: {stats['total_feedbacks']}
👍 Positive: {stats['positive_count']}
👎 Negative: {stats['negative_count']}
📤 Exported: {stats['exported_count']}

Model Performance:
"""
        perf = stats.get("model_performance")
        if perf:
            msg += f"""  Accuracy:  {perf['accuracy']:.1%}
  Precision: {perf['precision']:.1%}
  Recall:    {perf['recall']:.1%}
  Samples:   {perf['training_samples']}
"""
        else:
            msg += "  Not trained yet (need 10+ feedbacks)"
        
        QMessageBox.information(self, "AI Statistics", msg)

    def export_audio(self):
        if not self.audio_path or not self.candidates:
            return
        
        row = self.list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Error", "Please select a loop point first")
            return
        
        # 少なくとも1つのセクションが選択されているか確認
        if not (self.check_intro.isChecked() or self.check_loop.isChecked() or self.check_outro.isChecked()):
            QMessageBox.warning(self, "Error", "Please select at least one section to export")
            return
        
        c = self.candidates[row]
        
        # 保存先選択
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Edited Audio", 
            os.path.splitext(os.path.basename(self.audio_path))[0] + "_looped.mp3",
            "MP3 (*.mp3)"
        )
        if not save_path:
            return
        
        self.status_label.setText("EXPORTING...")
        self.btn_export.setEnabled(False)
        
        try:
            # オーディオ読み込み
            audio = AudioSegment.from_file(self.audio_path)
            
            # サンプル→ミリ秒変換
            start_ms = (c["s"] * 1000) // c["f"]
            end_ms = (c["e"] * 1000) // c["f"]
            
            # セクションを分割
            intro = audio[:start_ms]
            loop_section = audio[start_ms:end_ms]
            outro = audio[end_ms:]
            
            # 選択されたセクションを結合
            result = AudioSegment.empty()
            
            if self.check_intro.isChecked():
                result += intro
            
            if self.check_loop.isChecked():
                loop_count = self.spin_loop_count.value()
                result += (loop_section * loop_count)
            
            if self.check_outro.isChecked():
                result += outro
            
            # MP3出力
            result.export(save_path, format="mp3", bitrate="320k")
            
            # 成功時に自動的に👍フィードバックを記録
            if AI_AVAILABLE and self._ensure_ai_initialized():
                self._record_feedback(rating=1, source="export")
            
            duration_sec = len(result) / 1000
            self.status_label.setText(f"EXPORTED: {duration_sec:.1f}s")
            
            msg = f"Exported to:\n{save_path}\n\nDuration: {int(duration_sec//60)}:{int(duration_sec%60):02d}"
            if AI_AVAILABLE:
                msg += "\n\n✅ Marked as good loop for AI learning"
            
            QMessageBox.information(self, "Success", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            self.status_label.setText("EXPORT FAILED")
        
        finally:
            self.btn_export.setEnabled(True)

    def export_ogg_with_tags(self):
        """元のOGGファイルにLOOPSTART/LOOPLENGTHタグを付与してエクスポート"""
        if not self.audio_path or not self.candidates:
            return
        if not self.audio_path.lower().endswith(".ogg"):
            QMessageBox.warning(self, "Error", "OGGエクスポートはOGGファイルの入力時のみ使用できます")
            return
        if not MUTAGEN_AVAILABLE:
            QMessageBox.warning(self, "Error",
                "mutagen がインストールされていません。\n"
                "インストール: pip install mutagen")
            return

        row = self.list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Error", "ループポイントを選択してください")
            return

        c = self.candidates[row]
        loop_start = c["s"]
        loop_length = c["e"] - c["s"]

        # 保存先の選択
        default_name = os.path.splitext(os.path.basename(self.audio_path))[0] + "_looped.ogg"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save OGG with Loop Tags",
            default_name,
            "OGG (*.ogg)"
        )
        if not save_path:
            return

        self.status_label.setText("EXPORTING OGG...")
        self.btn_export_ogg.setEnabled(False)

        try:
            # 元ファイルをコピー（再エンコードなし）
            shutil.copy2(self.audio_path, save_path)

            # mutagenでVorbisコメントにLOOPSTART/LOOPLENGTHを書き込み
            ogg = OggVorbis(save_path)
            if ogg.tags is None:
                ogg.add_tags()
            ogg.tags["LOOPSTART"] = [str(loop_start)]
            ogg.tags["LOOPLENGTH"] = [str(loop_length)]
            ogg.save()

            self.status_label.setText("OGG EXPORTED")
            QMessageBox.information(
                self, "Success",
                f"OGGファイルをエクスポートしました:\n{save_path}\n\n"
                f"LOOPSTART  = {loop_start:,}\n"
                f"LOOPLENGTH = {loop_length:,}"
            )

            # 成功時にAIフィードバックを記録
            if AI_AVAILABLE and self._ensure_ai_initialized():
                self._record_feedback(rating=1, source="export")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            self.status_label.setText("EXPORT FAILED")
        finally:
            if self.audio_path and self.audio_path.lower().endswith(".ogg") and MUTAGEN_AVAILABLE:
                self.btn_export_ogg.setEnabled(True)

    def update_ui(self):
        if self.handle:
            curr = self.bass.BASS_ChannelGetPosition(self.handle, BASS_POS_BYTE)
            self.waveform.set_position(curr, self.current_loop_bytes[0], self.current_loop_bytes[1])

    def stop_audio(self):
        if self.handle: 
            self.bass.BASS_ChannelStop(self.handle)
            self.status_label.setText("PLAYBACK STOPPED")

    def closeEvent(self, event):
        if hasattr(self, 'bass'): self.bass.BASS_Free()
        event.accept()

if __name__ == "__main__":
    # 起動時にAI機能の有効性を表示
    print("=" * 50)
    print("Loop Profiler v2.0 - AI Enhanced")
    print("=" * 50)
    if AI_AVAILABLE:
        print("✅ AI機能: 有効")
    else:
        print("⚠️ AI機能: 無効(基本機能は動作します)")
        print("   有効化: pip install librosa scikit-learn")
    print("=" * 50)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = IntegratedLoopProfiler()
    win.show()
    sys.exit(app.exec())
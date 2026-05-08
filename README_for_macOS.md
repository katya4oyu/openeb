# Apple Silicon macOS で OpenEB をビルドして Viewer を動かす

このブランチでは、Apple Silicon macOS 向けに Devbox ベースの開発環境を用意しています。
Homebrew で C++ 依存関係を入れずに、Nix 経由で OpenEB の C++ SDK、samples、UI bindings、Python bindings をビルドします。

このブランチで追加した自作 Viewer は、OpenEB 標準の `metavision_sdk_ui` / OpenGL 経路を使いません。

- `utils/python/samples/pyside6_event_viewer.py`
- `utils/python/samples/opencv_event_viewer.py`

## 前提

- Apple Silicon Mac (`arm64` / `aarch64-darwin`)
- macOS
- 初回の Nix インストールで `sudo` できること
- ファイル再生用の RAW または HDF5 イベントファイル

Command Line Tools が未インストールの場合は入れてください。

```sh
xcode-select --install
```

## Nix と Devbox をインストールする

Devbox は内部で Nix package manager を使います。
このブランチでは、先に NixOS 公式インストーラで Nix を入れてから Devbox を使う前提にしています。

### 1. NixOS 公式インストーラで Nix を入れる

```sh
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install)
```

macOS では multi-user installation になるため、インストール中に `sudo` を求められます。
完了したら新しいターミナルを開き、確認します。

```sh
nix --version
```

### 2. Devbox を入れる

```sh
curl -fsSL https://get.jetify.com/devbox | bash
```

新しいターミナルを開き、確認します。

```sh
devbox version
```

補足:

- ここで入れるのは NixOS ではなく、Nix package manager です。
- インストーラスクリプトは NixOS project の `nixos.org` から取得します。
- Devbox を使うために Nix 言語を書く必要はありません。

参考:

- Devbox install docs: <https://www.jetify.com/docs/devbox/installing-devbox/>
- Nix install docs: <https://nixos.org/download/>

## OpenEB をビルドする

リポジトリのルートで実行します。

```sh
cd /path/to/openeb
devbox run configure
devbox run build
devbox run import-check
```

各コマンドの意味:

- `configure`: Python bindings 有効で CMake/Ninja を設定します。
- `build`: C++ libraries、samples、HAL plugins、Python bindings をビルドします。
- `import-check`: 主要な Python bindings を import できることを確認します。

この Devbox 環境には Core ML samples 用の Python 依存も含めています。
主な追加依存は PyTorch、TorchVision、PyTorch Lightning、Kornia、Numba、scikit-image です。

Apple Silicon で PyTorch MPS backend が見えているか確認するには、次を実行します。

```sh
devbox run -- python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

`True` と表示されれば、PyTorch から Metal Performance Shaders backend を利用できます。

## Core ML / MPS の動作確認

このブランチでは、`metavision_core_ml.video_to_event.GPUEventSimulator` が `mps` device の tensor を受け取れるようにしています。
CUDA kernel を Metal kernel に移植しているわけではなく、Numba kernel 部分は CPU 実装へ fallback し、入出力 tensor と状態を MPS device に戻します。
そのため、macOS で動かすための互換対応であり、CUDA と同等の GPU 高速化ではありません。

最小動作確認:

```sh
devbox run -- python -c 'import torch; from metavision_core_ml.video_to_event.gpu_simulator import GPUEventSimulator; device="mps"; sim=GPUEventSimulator(1, 4, 5).to(device); imgs=torch.randint(1,255,(4,5,3), dtype=torch.uint8, device=device); video_len=torch.tensor([3], device=device); ts=torch.tensor([[0.,1000.,2000.]], device=device); first=torch.tensor([1.], device=device); log=sim.dynamic_moving_average(imgs, video_len, ts, first); counts=sim.count_events(log, video_len, ts, first); vol=sim.event_volume(log, video_len, ts, first, 2); events=sim.get_events(log, video_len, ts, first); print(device, log.device, counts.device, vol.device, events.device, counts.shape, vol.shape, events.shape)'
```

期待される出力例:

```text
mps mps:0 mps:0 mps:0 mps:0 torch.Size([1, 4, 5]) torch.Size([1, 2, 4, 5]) torch.Size([N, 5])
```

`N` は生成されたイベント数なので、入力や乱数により変わります。

学習・デモ scripts では `--device mps` を指定できます。

```sh
devbox run -- python sdk/modules/core_ml/python/samples/viz_video_to_event_gpu_simulator/viz_video_to_event_gpu_simulator.py \
  /path/to/video_or_image_dataset \
  --device mps
```

MPS では `float64` や一部演算に制約があるため、学習 scripts は MPS 選択時に既定の `precision=16` を `32` に変更します。
CUDA 環境では従来通り `--device cuda` または `--device cuda:0` を使えます。

## Event-based Examples を起動する

Time Surface:

```sh
devbox run time-surface -- \
  -i /path/to/recording.raw \
  --accumulation-time 10000
```

Dummy Radar:

```sh
devbox run dummy-radar -- \
  -i /path/to/recording.raw \
  --accumulation-time 50000 \
  --nbins 8
```

Flow Inference:

```sh
devbox run flow-infer -- \
  /path/to/recording.raw \
  /path/to/e2v.ckpt \
  --device mps \
  --viz_input \
  --viz_gray
```

## PySide6 Viewer を起動する

Apple Silicon ではこちらを推奨します。
OpenGL を使わず、Qt の通常描画でイベントフレームを表示します。

RAW ファイルを開く場合:

```sh
devbox run -- python utils/python/samples/pyside6_event_viewer.py \
  -i /path/to/recording.raw
```

接続済みカメラをリアルタイム表示する場合は、`-i` を省略します。

```sh
devbox run -- python utils/python/samples/pyside6_event_viewer.py
```

## Spatter Tracking Lite を起動する

Analytics module なしで火花のような小さいイベント塊を簡易追跡するサンプルです。

```sh
devbox run spatter-lite -- \
  -i /path/to/sparklers.raw \
  --accumulation-time-us 50 \
  --cell-width 4 \
  --cell-height 4 \
  --activation-ths 2
```

HDF5 ファイルを開く場合:

```sh
devbox run -- python utils/python/samples/pyside6_event_viewer.py \
  -i /path/to/recording.hdf5
```

オプション例:

```sh
devbox run -- python utils/python/samples/pyside6_event_viewer.py \
  -i /path/to/recording.raw \
  --delta-t 10000 \
  --max-duration 5000000
```

操作:

- `Space` または `P`: 一時停止 / 再開
- `Q` または `Esc`: 終了

## OpenCV Viewer を起動する

軽い動作確認用の Viewer です。

```sh
devbox run -- python utils/python/samples/opencv_event_viewer.py \
  -i /path/to/recording.raw
```

接続済みカメラをリアルタイム表示する場合は、`-i` を省略します。

```sh
devbox run -- python utils/python/samples/opencv_event_viewer.py
```

タイミング指定の例:

```sh
devbox run -- python utils/python/samples/opencv_event_viewer.py \
  -i /path/to/recording.raw \
  --delta-t 10000 \
  --max-duration 5000000 \
  --realtime
```

操作:

- `Q` または `Esc`: 終了

注意:

macOS では、OpenCV と OpenEB 側の C++ OpenCV-linked bindings が同時に読み込まれたとき、
Objective-C class duplicate warning が出ることがあります。
通常利用では PySide6 Viewer を優先してください。

## なぜ OpenEB 標準の OpenGL Viewer を使わないのか

OpenEB 標準の UI samples は `metavision_sdk_ui` 経由で GLFW/GLEW/OpenGL を使います。
Apple Silicon macOS では、この経路で GLSL 互換性エラーが出ることがあります。

例:

```text
version '310' is not supported
syntax error: #version
One or more attached shaders not successfully compiled
```

このブランチの PySide6 / OpenCV Viewer は、イベントを NumPy でフレーム化し、
Qt または OpenCV window で表示するため、この OpenGL shader 経路を避けられます。

## トラブルシュート

### `devbox` が package を見つけられない、または install に失敗する

新しいターミナルで Nix が使えるか確認してください。

```sh
nix --version
```

その後、再実行します。

```sh
devbox run configure
```

### Python import に失敗する

bindings を作り直してください。

```sh
devbox run configure
devbox run build
devbox run import-check
```

### Viewer がファイルを開けない

入力ファイルが存在するか、RAW または HDF5 のイベントファイルか確認してください。

```sh
ls -lh /path/to/recording.raw
```

### カメラが見つからない

この README で説明している自作 Viewer は、ファイル再生と live camera の両方に対応しています。
ファイルを再生する場合は `-i /path/to/file.raw` または `-i /path/to/file.hdf5` を指定してください。
接続済みカメラを開く場合は `-i` を省略してください。

Live camera support は、USB 権限や実機ごとの検証が別途必要です。

このブランチでは、CenturyArks SilkyEvCam Gen3.1 (`USB Vendor ID: 0x31f7`, `USB Product ID: 0x0002`) を
OpenEB の Treuzell discovery 対象に追加しています。

まず OpenEB HAL から見えているか確認します。

```sh
devbox run camera-list
```

詳細ログを見る場合:

```sh
devbox run camera-trace
```

macOS の USB 情報として見えているか確認する場合:

```sh
devbox run usb-list
```

`usb-list` には出るが `camera-list` が `No device found` の場合、OpenEB HAL plugin までは読めているが、
そのカメラの USB interface descriptor または Treuzell protocol 互換性で弾かれている可能性があります。
この場合は `camera-trace` の出力を確認してください。

## 開発メモ

Devbox 環境は以下で定義しています。

- `devbox.json`
- `devbox.lock`

ローカル生成物は git 管理しません。

- `.devbox/`
- `build/`

Devbox の依存を変えた後は、再度 configure/build してください。

```sh
devbox run configure
devbox run build
```

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

この環境では Core ML 依存、たとえば PyTorch は入れていません。

## PySide6 Viewer を起動する

Apple Silicon ではこちらを推奨します。
OpenGL を使わず、Qt の通常描画でイベントフレームを表示します。

RAW ファイルを開く場合:

```sh
devbox run -- python utils/python/samples/pyside6_event_viewer.py \
  -i /path/to/recording.raw
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

この README で説明している自作 Viewer はファイル再生向けです。
`-i /path/to/file.raw` または `-i /path/to/file.hdf5` を指定してください。

Live camera support は、USB 権限や実機ごとの検証が別途必要です。

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

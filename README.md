# Ternary-Bonsai-8B 推論テスト

Hugging Face で公開されている 1.58 ビットの 3 値量子化モデル `onnx-community/Ternary-Bonsai-8B-ONNX` をローカルで動作させるための検証用リポジトリです。

パッケージマネージャとして `uv` を使用しており、環境構築から実行まで自動的に行われます。

このリポジトリでは、ダウンロードの手軽さからまず Hugging Face 公開の q2 packed ONNX を使う方針を前提にしています。ただし、現状の ONNX Runtime 環境ではこの q2 モデルをそのまま実行できないため、実運用上は一度ローカルで q8 ONNX に変換してから推論に使っています。

## 動作環境と実行方法

`pyproject.toml` に OS 別の条件分岐 (`sys_platform == 'win32'`) を設定しているため、Windows 環境で実行したときのみ GPU アクセラレーション用の `onnxruntime-directml` が自動的にインストールされます。

### Safetensors 環境 (CPU 評価)

packed ONNX ではなく、Hugging Face の safetensors 版 `prism-ml/Ternary-Bonsai-8B-unpacked` をそのまま CPU で評価するスクリプトです。モデルのロード時間、生成時間、tokens/sec を確認できます。

```bash
uv run run_safetensors.py --max-new-tokens 32
```

必要なら dtype も切り替えられます。

```bash
uv run run_safetensors.py --dtype float32 --max-new-tokens 16
```

### Windows 環境 (DirectML による GPU アクセラレーション)

Windows PC (Radeon, Intel, GeForce など) で内蔵または外部 GPU の VRAM を活用して推論を行います。ベースとしては入手しやすい q2 packed ONNX を使いますが、この q2 モデルは現状そのままでは実行できないため、初回実行時にローカルで q8 ONNX に展開してから読み込みます。そのぶん初回は時間とディスク容量が追加で必要です。

```bash
uv run run_directml.py
```

### Linux 環境 (CPU 推論)

Linux (GPU なし) の環境では、標準の CPU 実行プロバイダが使用されます。

※現在の ONNX Runtime (CPU) はこのモデル特有の特殊な量子化演算をサポートしていないため、事前に 8 ビットに展開する変換スクリプトを挟むか、または他の GGUF 形式のモデルを利用することを推奨します。

```bash
uv run run_q2_to_q8.py
```

## 計測結果

Windows 上で `compare.bat 10` を実行し、safetensors 由来の ONNX である fp32 export、fp16 export、fp8 conversion を同一条件で比較した結果です。

| モデル | Library load | Model load | Generation | Total |
| --- | ---: | ---: | ---: | ---: |
| fp32 export | 15.80s | 568.04s | 12.01s | 595.99s |
| fp16 export | 11.28s | 190.35s | 21.15s | 223.90s |
| fp8 conversion | 11.35s | 124.96s | 22.70s | 159.80s |

この条件では、生成品質の出だしは 3 つとも同等でした。速度面では、fp16 export と fp8 conversion は生成そのものは fp32 よりやや遅いものの、モデルロード時間が大幅に短く、全体時間は短くなりました。

### 10 トークン比較

まず safetensors 原本を CPU で計測し、その後に `compare.bat 10` で ONNX 5 モデル × 2 プロバイダの全組み合わせを計測した結果です。`q2->fp8 conversion` の 2 行は今回追加した変換経路の実測値です。

| 実行条件         | Library load | Model load | Generation | Total |
| ------------------------- | -----: | ------: | ------: | ------: |
| safetensors / CPU         |  3.51s |   3.13s |  17.30s |  23.95s |
| fp32 export / CPU         |  3.33s | 569.20s |  11.27s | 583.95s |
| fp32 export / DirectML    |  7.66s | 119.50s |  19.88s | 147.19s |
| fp16 export / CPU         |  7.56s | 196.12s |  14.58s | 219.32s |
| fp16 export / DirectML    | 15.80s | 148.99s |   6.70s | 171.50s |
| fp8 conversion / CPU      |  3.48s | 193.46s |  22.29s | 220.18s |
| fp8 conversion / DirectML | 18.87s | 136.39s |   6.75s | 162.02s |
| q2->fp8 conversion / CPU  |  3.58s | 147.76s |  43.67s | 195.90s |
| q2->fp8 conversion / DirectML |  3.51s |  67.34s |   3.52s |  74.39s |
| q8 conversion / CPU       |  3.30s |   8.59s | 281.27s | 293.17s |
| q8 conversion / DirectML  |  3.39s |   8.12s |  10.47s |  22.00s |

`q8 conversion` では Optimum から legacy ONNX と `position_ids` 不在に関する警告が出ています。

`safetensors / CPU` はモデルロードが最も軽く、全体時間も `q8 conversion / DirectML` に近い値でした。一方で ONNX 系はモデルごとのレイアウトや実行プロバイダ依存の差が大きく、単純な速度比較だけでは評価しきれません。

品質面では、`fp32` / `fp16` / `fp8` は CPU と DirectML のどちらでも日本語の自然な出だしでした。今回追加した `q2->fp8 conversion` は CPU では自然な出だしを維持した一方、DirectML では生成が崩れました。`q8 conversion / DirectML` でも同様の崩れがあるため、現時点では q2 由来のローカル変換モデルを DirectML で使う場合は品質面の注意が必要です。速度最優先なら `q8 conversion / DirectML` や `q2->fp8 conversion / DirectML` は魅力がありますが、実用面ではまだ追加検証が必要です。

## ONNX エクスポート方法

再エクスポート元には Hugging Face の safetensors 版 `prism-ml/Ternary-Bonsai-8B-unpacked` を使用します。

### fp32 export

```bash
uv run optimum-cli export onnx -m prism-ml/Ternary-Bonsai-8B-unpacked model_fp32 --task text-generation-with-past --pad_token_id 151643 --dtype fp32
```

### fp16 export

```bash
uv run optimum-cli export onnx -m prism-ml/Ternary-Bonsai-8B-unpacked model_fp16 --task text-generation-with-past --pad_token_id 151643 --dtype fp16
```

### fp8 conversion

`optimum-cli export onnx` は `fp8` を直接サポートしていないため、まず fp16 ONNX を作成し、その後に [convert_fp8.py](convert_fp8.py) で後処理します。

```bash
uv run python convert_fp8.py --source-model-path model_fp16/model.onnx --output-model-path model_fp8/model.onnx
```

### エクスポート結果の比較

エクスポート後は次のバッチで全組み合わせを同条件比較できます。

```bat
compare.bat 10
```

`compare.bat` は次の 10 組を順番に実行します。

- fp32 export / CPU
- fp32 export / DirectML
- fp16 export / CPU
- fp16 export / DirectML
- fp8 conversion / CPU
- fp8 conversion / DirectML
- q2->fp8 conversion / CPU
- q2->fp8 conversion / DirectML
- q8 conversion / CPU
- q8 conversion / DirectML

### 補足

- `accelerate` が入っていないと、weight deduplication の確認で警告が出ます。
- 現在の `optimum-cli export onnx` が受け付ける dtype は `fp32`、`fp16`、`bf16` です。
- `fp8 conversion` は `optimum-cli export onnx` から直接は生成していません。`--dtype fp8` は invalid choice で失敗するため、fp16 ONNX からの後処理変換で生成しています。

## モデル形式の違い

このリポジトリには、見た目はどれも ONNX ですが、中身の表現が異なるモデルが混在しています。

### safetensors / fp32 export / fp16 export / fp8 conversion

`prism-ml/Ternary-Bonsai-8B-unpacked` は名前の通り unpacked 版で、保存形式としては dense な重みです。モデルの出自は ternary ですが、保存されている safetensors は packed ternary のままではなく、通常の dense tensor として展開されています。

そのため、ここから作る fp32 / fp16 / fp8 ONNX も dense tensor です。この種のモデルでは、ファイルサイズは主に `要素数 × 1 要素あたりのバイト数` で決まり、値が疎か密か、あるいは元モデルが ternary 由来かどうかではほとんど変わりません。

- fp32: 1 要素 4 byte
- fp16: 1 要素 2 byte
- fp8: 1 要素 1 byte

したがって、fp16 から fp8 への変換は理論上ほぼ半減が上限です。

### model_q2_to_q8 の 8-bit 変換

一方、`model_q2_to_q8` は `onnx-community/Ternary-Bonsai-8B-ONNX` の 2-bit packed ONNX から [convert_q8.py](convert_q8.py) で生成したものです。こちらは dense fp16 から作ったわけではなく、`MatMulNBits` が解釈する packed 量子化表現から出発しています。

この変換を採用している理由は、q2 packed ONNX は配布サイズが小さく入手しやすい一方で、現在の実行環境ではそのまま推論に使えないためです。まず q2 をダウンロードし、ローカルで q8 に展開して実行可能な形へ寄せる、というのがこのリポジトリでの実用上のワークフローです。

`convert_q8.py` はこの packed 2-bit 重みを ONNX Runtime が扱える q8 形へ展開しています。そのため `model_q2_to_q8` は safetensors 由来の fp16 / fp8 export 系とはサイズの意味が異なり、より小さいサイズになります。

#### 警告と品質低下の切り分け

`model_q2_to_q8` は元の q2 ONNX のグラフ構造をほぼそのまま引き継いでいるため、`position_ids` 入力を持たず、代わりに `num_logits_to_keep` 入力を持っています。このため Optimum から legacy ONNX と `position_ids` 不在に関する警告が出ます。

ただし、この警告自体が直ちに生成崩れの原因とは限りません。実際には `q8 conversion / CPU` では日本語の自然な出だしが出ており、生成品質の大きな崩れは `q8 conversion / DirectML` でのみ確認されています。そのため、現時点では q2 から q8 への変換そのものよりも、変換後 ONNX を DirectML で実行したときの数値安定性または演算互換性の問題である可能性が高いと考えられます。

#### 変換方法

`model_q2_to_q8` を使う場合は、先に次を実行して手動で変換してください。

変換元の q2 ONNX がまだ `model_cache` に無い場合は、先に次でダウンロードします。

```bash
uv run python download_q2.py
```

```bash
uv run python convert_q8.py
```

変換元は `model_cache/Ternary-Bonsai-8B-ONNX/onnx/model_q2f16.onnx`、変換先は `model_q2_to_q8/model.onnx` です。外部データは `model_q2_to_q8/model.onnx_data` として保存され、tokenizer や config も同じディレクトリにコピーされます。

### model_q2_to_fp8 の fp8 変換

q8 変換は CPU では極端に遅く、DirectML では生成品質が崩れるため、別経路として q2 packed ONNX から dense fp8 ONNX を生成するスクリプトも追加しています。

これは q2 の `MatMulNBits` をそのまま fp8 化するのではなく、いったん各重みをブロック量子化表現から実数値へ復元し、それを fp8 で保存した dense `MatMul` に組み替える方式です。実行時には `Cast` で fp16 に戻して使うため、実質的には safetensors 由来の fp8 export に近い挙動を狙うものです。

変換元の q2 ONNX がまだ `model_cache` に無い場合は、先に次でダウンロードします。

```bash
uv run python download_q2.py
```

その後、次で q2 から fp8 へ変換します。

```bash
uv run python convert_q2_to_fp8.py
```

生成物は `model_q2_to_fp8/model.onnx` と `model_q2_to_fp8/model.onnx_data` です。実行時は次のように指定できます。

```bash
uv run run_directml.py --model-source q2fp8
```

この方法はダウンロード元として q2 を使える一方で、変換後の重みは dense fp8 になるため、ローカルの生成物サイズはおおむね 7.6GB 規模になります。つまり、配布サイズの軽さは維持できますが、変換後のディスク使用量と変換時間は fp8 export 系に近づきます。

### まとめ

- fp32 / fp16 / fp8 export 系: safetensors 由来の dense tensor
- `model_q2_to_q8`: packed q2 ONNX 由来の q8 展開モデル
- `model_q2_to_fp8`: packed q2 ONNX を dense fp8 `MatMul` へ組み替えたモデル
- そのため、`model_q2_to_q8` と fp16 / fp8 export 系はサイズや圧縮率を単純比較できません

## 注意事項

- Windows 環境で `run_directml.py` を実行すると、自動的に Hugging Face から約 1.75GB のモデルデータがダウンロードされます。
- これは q2 packed ONNX を取得しているためダウンロード自体は比較的軽量ですが、そのままでは実行できないため、続けて q8 へのローカル変換が必要です。
- `model_q2_to_q8` を作成する場合は、`uv run python convert_q8.py` の実行でさらに約 7GB 規模の 8-bit 変換済みモデルが生成されます。
- `model_q2_to_fp8` を作成する場合も、`uv run python convert_q2_to_fp8.py` の実行で約 7.6GB 規模の dense fp8 モデルが生成されます。
- Windows で DirectML を使う構成では、通常版の `onnxruntime` と `onnxruntime-directml` を同居させないでください。通常版が優先して読み込まれると `DmlExecutionProvider` が見えなくなります。
- `onnxruntime-directml` はバージョンによって DML provider の同梱状況が異なるため、このリポジトリでは動作確認済みの版に固定しています。
- Windows では Hugging Face キャッシュ内のシンボリックリンクを ONNX Runtime がうまく扱えないため、`run_directml.py` は必要ファイルを `model_cache` 配下に実体コピーしてから読み込みます。
- `uv` が依存関係の解決と仮想環境 (`.venv`) の作成を自動的に行うため、事前の `pip install` は不要です。

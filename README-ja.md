# Ternary-Bonsai-8B 推論テスト

Hugging Face で公開されている 1.58 ビットの 3 値量子化モデル [onnx-community/Ternary-Bonsai-8B-ONNX](https://huggingface.co/onnx-community/Ternary-Bonsai-8B-ONNX) をローカルで動作させるための検証用リポジトリです。パッケージマネージャには `uv` を使用しており、依存関係の解決と仮想環境の作成が自動的に行われます。

> [英語版](README.md) | **日本語版**

## クイックスタート

推奨構成は `onnx_q2_to_q4 / CPU` です。q2 packed ONNX をダウンロードし、ローカルで q4 に変換してから推論します。

```bash
uv run download_q2.py
uv run convert_q2.py --target-format q4
uv run run_onnx.py --provider cpu --model-source q4
```

## 実行方法

### ONNX モデル ([run_onnx.py](run_onnx.py))

| オプション | 説明 |
|-----------|------|
| `--provider dml` | DirectML (Windows GPU、デフォルト) |
| `--provider cpu` | CPU 実行プロバイダ |
| `--model-source auto` | 利用可能なモデルを自動選択 (デフォルト) |
| `--model-source q4` | q2 → q4 変換モデル |
| `--model-source q8` | q2 → q8 変換モデル |
| `--model-source q2fp8` | q2 → dense fp8 変換モデル |
| `--model-source fp8` | safetensors 由来 fp8 エクスポート |
| `--model-source fp16` | safetensors 由来 fp16 エクスポート |
| `--model-source fp32` | safetensors 由来 fp32 エクスポート |
| `--max-new-tokens N` | 生成トークン数 (デフォルト 10) |
| `--prompt "..."` | 入力プロンプト |

```bash
uv run run_onnx.py --provider cpu --model-source q4 --max-new-tokens 32
```

### Safetensors モデル ([run_safetensors.py](run_safetensors.py))

Hugging Face の safetensors 版 `prism-ml/Ternary-Bonsai-8B-unpacked` を CPU で直接推論します。

```bash
uv run run_safetensors.py --max-new-tokens 32
uv run run_safetensors.py --dtype float32 --max-new-tokens 16
```

### q2 packed ONNX 直接実行 ([run_q2.py](run_q2.py))

現状の ONNX Runtime では失敗することを確認するためのスクリプトです。

```bash
uv run run_q2.py
```

### PyTorch ネイティブ 3 値演算 ([run_q2_pytorch.py](run_q2_pytorch.py))

q2 ONNX ファイルから全重みを直接抽出し、block-scaled な 3 値整数演算を純粋な PyTorch で実装して推論します。追加のモデルダウンロードは不要です。実用目的ではなく、3 値量子化の仕組みを技術的に調査するためのスクリプトです。

```bash
uv run run_q2_pytorch.py
```

線形層の重みが真に 3 値であることを確認できます。値 −1・0・+1 のみ使用されており、それぞれ約 33% ずつ出現します。ONNX の重みフォーマットの詳細は [ONNX_Q2.md](ONNX_Q2.md) を参照してください。

### 一括比較 ([compare.bat](compare.bat))

全 13 条件 (safetensors + 各 ONNX × CPU/DirectML) を順次実行します。

```bat
compare.bat 10
```

## モデル形式

このリポジトリでは出自の異なる 2 系統の形式を扱っています。

### safetensors 由来 (標準 ONNX グラフ)

`prism-ml/Ternary-Bonsai-8B-unpacked` の dense 重みから `optimum-cli export onnx` で生成します。グラフ内の演算はすべて標準 ONNX op です。

| 形式 | 1 要素 | 生成方法 |
|-------|-------:|---------|
| `onnx_fp32` | 4 byte | `optimum-cli export onnx --dtype fp32` |
| `onnx_fp16` | 2 byte | `optimum-cli export onnx --dtype fp16` |
| `onnx_fp8` | 1 byte | fp16 ONNX を [convert_fp8.py](convert_fp8.py) で後処理 |

### q2 packed ONNX 由来 (`com.microsoft` カスタム op グラフ)

`onnx-community/Ternary-Bonsai-8B-ONNX` の 2-bit packed ONNX から変換します。グラフ構造に `GroupQueryAttention` などの `com.microsoft` カスタム op が残るため、safetensors 系とはグラフの構造自体が異なります。

| 形式 | 変換スクリプト | 形式 |
|-------|-------------|------|
| `onnx_q2_to_q4` | [convert_q2.py](convert_q2.py) `--target-format q4` | `MatMulNBits` q4 |
| `onnx_q2_to_q8` | [convert_q2.py](convert_q2.py) `--target-format q8` | `MatMulNBits` q8 |
| `onnx_q2_to_fp8` | [convert_q2_to_fp8.py](convert_q2_to_fp8.py) | dense fp8 `MatMul` |

`onnx_q2_to_fp8` は重みを dense fp8 に展開するため、重み自体は `onnx_fp8` と同等ですが、attention・normalization・embedding 部分のグラフ構造が異なります。この違いが DirectML での挙動差の原因です ([ONNX.md](ONNX.md))。

## 変換手順

### q2 packed ONNX のダウンロード

```bash
uv run download_q2.py
```

### q2 → q4 / q8 変換

```bash
uv run convert_q2.py --target-format q4
uv run convert_q2.py --target-format q8
```

### q2 → dense fp8 変換

```bash
uv run convert_q2_to_fp8.py
```

### safetensors → ONNX エクスポート

```bash
uv run optimum-cli export onnx -m prism-ml/Ternary-Bonsai-8B-unpacked onnx_fp32 --task text-generation-with-past --pad_token_id 151643 --dtype fp32
uv run optimum-cli export onnx -m prism-ml/Ternary-Bonsai-8B-unpacked onnx_fp16 --task text-generation-with-past --pad_token_id 151643 --dtype fp16
uv run convert_fp8.py --source-model-path onnx_fp16/model.onnx --output-model-path onnx_fp8/model.onnx
```

`optimum-cli export onnx` が受け付ける dtype は `fp32`、`fp16`、`bf16` です。`--dtype fp8` は使えないため、fp8 は fp16 ONNX からの後処理で生成します。

## ベンチマーク

`compare.bat 10` による計測結果です。「ロード」にはトークナイザーとモデルの読み込みに加えてウォームアップ (1 トークン入出力) を含みます。モデルは HDD に保存しているため読み込みが遅いですが、2 回目 (DirectML) は OS のファイルキャッシュにより高速化されます。

**プロンプト**：AIの未来について考えてください。

| 形式 | 方式 | ロード | 推論 | 出力 |
|---|---|---:|---:|---|
| `safetensors_fp16` | CPU | 170.30s | 9.82s | AIの未来について考えるとき、いくつかの重要な |
| `onnx_fp16` | CPU | 213.59s | 9.59s | AIの未来について考えるとき、いくつかの重要な |
| `onnx_fp16` | DirectML | 149.75s | 4.75s | AIの未来について考えるとき、いくつかの重要な |
| `onnx_fp8` | CPU | 206.62s | 11.09s | AIの未来について考えるときは、いくつかの重要な |
| `onnx_fp8` | DirectML | 137.75s | 5.30s | AIの未来について考えるときは、いくつかの重要な |
| `onnx_q2_to_fp8` | CPU | 235.20s | 10.18s | AIの未来について考えるときは、いくつかの重要な |
| `onnx_q2_to_fp8` | DirectML | 131.84s | 2.48s | AI inki authoritative undert particles replicate faculty guess order |
| `onnx_q2_to_q8` | CPU | 106.68s | 199.62s | AIの未来について考えるとき、いくつかの重要な |
| `onnx_q2_to_q8` | DirectML | 8.65s | 7.02s | AI Nation informed simply Adult Hobby pis Agents contributing |
| `onnx_q2_to_q4` | CPU | 37.59s | 6.46s | AIの未来について考えるとき、いくつかの重要な |
| `onnx_q2_to_q4` | DirectML | 5.96s | 1.12s | AI Nation informed simply formats teacher norm Sh Earth |

1000 トークンまで生成した結果: [SAMPLE.md](SAMPLE.md)

### 推奨構成

**`onnx_q2_to_q4 / CPU`** が総合的に最もバランスが良い選択です。

- Load 37.59s / Generation 6.46s と十分に軽量
- 出力品質は自然で安定
- 2 回目以降は OS のファイルキャッシュにより大幅に高速化
- CPU 単体で `onnx_fp8 / DirectML` に迫る生成速度
- q2 ダウンロード (約 1.75GB) から変換できるため入手も容易

### DirectML の制限

q2 由来の 3 モデル (`onnx_q2_to_fp8`、`onnx_q2_to_q4`、`onnx_q2_to_q8`) は DirectML で生成が崩れます。原因は重みの精度ではなく、q2 グラフが引き継ぐ `com.microsoft` カスタム op (主に `GroupQueryAttention`) を DirectML が正しく実行できないことにあります。CPU では同じモデルが正常に動作します。詳細な調査結果は [ONNX.md](ONNX.md) を参照してください。

DirectML で安定した品質が必要な場合は、safetensors 由来の `onnx_fp16` または `onnx_fp8` を使ってください。

### その他の所見

- `onnx_fp16` / `onnx_fp8` はいずれも CPU・DirectML とも品質安定。ただし HDD ではロード時間が長い。
- `onnx_q2_to_q8 / CPU` は品質は崩れないが Generation 199.62s と極端に遅い。
- `onnx_q2_to_q8` は Optimum から legacy ONNX と `position_ids` 不在に関する警告が出る。これはグラフに `num_logits_to_keep` 入力があり `position_ids` 入力がないため。

## 注意事項

- Windows で `run_onnx.py` を実行すると、Hugging Face から約 1.75GB の q2 packed ONNX が自動ダウンロードされます。
- q4 変換で追加約 3.5GB、q8 変換で約 7GB、fp8 変換で約 7.6GB のディスク容量が必要です。
- Windows で DirectML を使う場合、`onnxruntime` と `onnxruntime-directml` を同居させないでください。通常版が優先して読み込まれると `DmlExecutionProvider` が見えなくなります。
- `onnxruntime-directml` はバージョンにより DML provider の同梱状況が異なるため、動作確認済みの版に固定しています。
- Windows では Hugging Face キャッシュ内のシンボリックリンクを ONNX Runtime が扱えないため、[download_q2.py](download_q2.py) はファイルを `onnx_q2` 配下に実体化します。
- `pyproject.toml` の `sys_platform == 'win32'` 条件により、Windows のみ `onnxruntime-directml` が自動インストールされます。

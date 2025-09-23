# gov-minutes-rag

日本の各省庁が公開している審議会・委員会の議事録を横断検索するための RAG（Retrieval-Augmented Generation）基盤です。正規化・埋め込み・API までをワンストップで扱います。

## 機能概要
- 正規化スクリプト（`scripts/normalize.py`）: DB や `sample/` 配下から議事録メタデータを抽出し、`meeting_metadata` に UPSERT します
- ベクトル化スクリプト（`scripts/embed.py`）: 本文をチャンク・要約して埋め込みを生成し、`meeting_chunks` と `chunks_summary` に保存します
- FastAPI（`api/main.py`）: 要約ベクトルのみを対象にした `/summary_search` エンドポイントを提供します

## スキーマ適用
`schema/rag_tables.sql` に DDL をまとめています。PostgreSQL + pgvector 拡張を前提に以下を実行してください。

```
psql $DATABASE_URL -f schema/rag_tables.sql
```

## セットアップ
1. 必要に応じて venv / Poetry を用意
2. `.env.example` をコピーして接続先などの環境変数を設定
3. 上記のスキーマ適用を実行（pgvector / pg_trgm 拡張が有効化されます）

```bash
cp .env.example .env
psql postgresql://user:pass@localhost:5432/gov_minutes -f schema/rag_tables.sql
```

### Docker Compose での起動
PostgreSQL（pgvector）と API をまとめて起動する場合:

```bash
cd docker
docker compose up -d
docker compose exec db psql -U gov -d gov_minutes -f ../schema/rag_tables.sql
```

API は `http://localhost:8000` で公開され、要約検索が利用できます。

## 正規化とメタデータ登録
`crawled_pages` テーブルや `sample/` フォルダから未処理の URL を抽出し、`meeting_metadata` に UPSERT します。

```bash
python scripts/normalize.py --source both --limit 100
# --source db | sample | both
```

メモ:
- URL をキーに UPSERT するため再実行しても安全
- PDF は `PyPDF2` が存在すれば抽出、無ければ空文字（警告）
- 会議名や日付などはヒューリスティックで抽出し、失敗した項目は空 JSON

## チャンク化と埋め込み生成
`meeting_metadata` から未埋め込みの議事録を選び、本文をチャンク化して `meeting_chunks`、要約を `chunks_summary` に格納します。

```bash
python scripts/embed.py --limit-docs 50
```

## Embedding プロバイダ
`.env` で指定できます。

- `EMBEDDING_PROVIDER=openai`（`OPENAI_API_KEY` と `EMBEDDING_MODEL=text-embedding-3-small` などを指定）
- `EMBEDDING_PROVIDER=local`（`sentence-transformers` を利用。未インストール時はハッシュ擬似ベクトルにフォールバック）

OpenAI 実行時は指数バックオフでリトライします。

## API の起動
ローカルで API を立ち上げる場合:

```bash
uvicorn api.main:app --reload --port 8000
```

Docker Compose で起動した場合は `docker compose up -d` 後に自動で `uvicorn` が開始されます。

### `/summary_search`
要約ベクトルのみを対象とした検索エンドポイントです。`top_k` は 1〜100、`ministry` で府省庁をフィルタできます。

```bash
curl -X POST http://localhost:8000/summary_search \
  -H "Content-Type: application/json" \
  -d '{"query":"感染症対策", "top_k":5, "ministry":"厚生労働省"}'
```

レスポンス例:

```json
[
  {
    "url": "https://example.go.jp/...",
    "council_name": "厚生科学審議会...",
    "date": "2025-07-23",
    "summary": "...",
    "score": 0.83
  }
]
```

## ディレクトリ構成
```
gov-minutes-rag/
  sample/
  schema/
    rag_tables.sql
  scripts/
    normalize.py
    embed.py
  api/
    main.py
  docker/
    Dockerfile
    docker-compose.yml
  .env.example
  README.md
```

## 注意事項と再実行のコツ
- `meeting_metadata.url` にユニーク制約を設定し、既存 URL は UPSERT で更新
- `meeting_chunks` / `chunks_summary` はドキュメント単位で再生成（既存削除→再挿入）して重複を回避
- pgvector の HNSW インデックス（cosine）と `pg_trgm` インデックスを用意すると高速化できます

## よくあるトラブル
- `openai` / `sentence-transformers` が未インストール: ローカル擬似ベクトルにフォールバック（精度は低下）
- PDF 抽出が空になる: `PyPDF2` をインストール、または別抽出器の導入を検討
- スキーマ未適用: `schema/rag_tables.sql` を必ず実行してください

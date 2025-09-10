# gov-minutes-rag

日本の各省庁が公開している審議会・委員会の議事録を、RAG（Retrieval-Augmented Generation）で横断検索するための最小実装です。

## 機能概要
- 正規化スクリプト（`scripts/normalize.py`）で、以下2系統のソースから議事録メタ情報を生成してDBにUPSERT保存します。
  - DB: `crawled_pages`（URL, HTMLタイトル, content(BYTEA)のHTML/PDFなど）
  - ローカル: `sample/` 配下のPDF/HTML/TXT
- ベクトル化スクリプト（`scripts/embed.py`）で、本文をチャンク化し、埋め込みを生成して`meeting_chunks`へ保存します。
- FastAPI（`api/main.py`）で、クエリをベクトル化してpgvector類似検索を実行する`/search`エンドポイントを提供します。

## スキーマ
`schema/rag_tables.sql` にDDLをまとめています（PostgreSQL + pgvector）。

```
psql $DATABASE_URL -f schema/rag_tables.sql
```

## セットアップ
1. Python/Poetryやvenvを用意（任意）
2. `.env` を作成（`.env.example`をコピーして編集）
3. DBにDDL適用（pgvectorとpg_trgmが有効化されます）

```bash
cp .env.example .env
psql postgresql://user:pass@localhost:5432/gov_minutes -f schema/rag_tables.sql
```

Docker ComposeでPostgres+pgvectorを起動する場合:

```bash
cd docker
docker compose up -d db
psql postgresql://gov:govpass@localhost:5432/gov_minutes -f ../schema/rag_tables.sql
```

## 正規化（メタ情報登録）
`crawled_pages`テーブル（別システムで管理）から未処理URLを抽出、または`sample/`フォルダから読み込み、`meeting_metadata`にUPSERTします。

```bash
python scripts/normalize.py --source both --limit 100
# --source db | sample | both
```

メモ:
- URL一意でUPSERTし、`doc_id`は`uuid5(NAMESPACE_URL, url)`で安定化。
- PDFは`PyPDF2`がある場合に抽出、無ければ空文字（警告）。HTMLは`BeautifulSoup`があればより精度良く抽出。
- 会議名・会議番号・日付（和暦/全角対応）・場所・出席者・議題をヒューリスティックに抽出、難しい場合は空JSON。

## チャンク化＆ベクトル化
`meeting_metadata`で未埋め込みの文書を選び、本文（`discussion_text`）をチャンク化（約1200-2000文字、オーバーラップ200字）して埋め込みを生成し、`meeting_chunks`へ保存します。

```bash
python scripts/embed.py --limit-docs 50
```

Embeddingプロバイダは `.env` で切り替え可能:

- `EMBEDDING_PROVIDER=openai` + `OPENAI_API_KEY`（`EMBEDDING_MODEL=text-embedding-3-small`等）
- `EMBEDDING_PROVIDER=local`（`sentence-transformers`があればローカルモデル、無ければハッシュ擬似ベクトル）

レート制限に対して指数バックオフを実装済み（OpenAI）。

## API の起動

```bash
uvicorn api.main:app --reload --port 8000
```

検索例:

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"感染症 審議会", "top_k":5, "ministry":"厚生労働省"}'
```

レスポンス例:

```json
[
  {"url":"https://example.go.jp/...","council_name":"厚生科学審議会 ...","date":"2025-07-23","chunk_text":"...","score":0.83}
]
```

## ディレクトリ構成

```
gov-minutes-rag/
  sample/
    (PDF/HTMLを配置)
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

## 注意事項・再実行安全
- `meeting_metadata.url`にユニーク制約、既存URLはUPSERTで更新。
- `meeting_chunks`はドキュメント毎に再生成（既存削除→挿入）で重複を避けます。
- `pgvector`のHNSWインデックス（cosine）と`pg_trgm`（ハイブリッド用途）を用意。

## よくあるトラブル
- `openai` / `sentence-transformers` が未インストール: ローカル擬似ベクトルにフォールバックします（精度は出ません）。
- PDF抽出が空: `PyPDF2`をインストール、または別の抽出器の導入を検討。
- スキーマ未適用: `schema/rag_tables.sql`を必ず実行してください。


import requests
import csv

BASE = "http://localhost:8000"

def main():
    # クエリCSVを読み込む
    queries = []
    with open("queries.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append((row["query_id"], row["query_text"]))

    # 結果保存用CSVを開く
    with open("search_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "query_text", "run", "rank",
            "doc_id", "url", "text_or_summary", "score"
        ])

        for qid, query in queries:
            print(f"Running query {qid}: {query}")
            try:
                resp = requests.post(
                    f"{BASE}/debug/compare_search",
                    json={"query": query, "top_k": 20},
                    timeout=60
                )
                resp.raise_for_status()
                data = resp.json()

                # chunks と summaries 両方保存
                for run in ["chunks", "summaries"]:
                    for rank, hit in enumerate(data[run], start=1):
                        # テキスト or サマリの先頭100文字だけ
                        text_snippet = (hit.get("text") or hit.get("summary") or "")
                        snippet = text_snippet[:50]
                        writer.writerow([
                            qid,
                            query,
                            run,
                            rank,
                            hit["doc_id"],
                            hit["url"],
                            snippet,
                            hit["score"]
                        ])
            except Exception as e:
                print(f"Query {qid} failed: {e}")

if __name__ == "__main__":
    main()

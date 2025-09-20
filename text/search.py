import os
import requests
import fugashi
import argparse

# 形態素解析 (MeCab/fugashi)
tagger = fugashi.Tagger()

def tokenize_query(query: str) -> str:
    return " ".join(word.surface for word in tagger(query))

def search_meilisearch(query: str, limit: int = 20):
    url = os.getenv("MEILI_URL", "http://127.0.0.1:7700") + "/indexes/meetings/search"
    headers = {
        "Authorization": f"Bearer " + os.getenv("MEILI_KEY", "MASTER_KEY"),
        "Content-Type": "application/json",
    }
    tokenized = tokenize_query(query)

    res = requests.post(
        url,
        headers=headers,
        json={
            "q": tokenized,
            "limit": limit,
            "attributesToRetrieve": ["url"]
        }
    )
    res.raise_for_status()
    hits = res.json().get("hits", [])
    return [(h["url"], h.get("_rankingScore")) for h in hits if "url" in h]

def main():
    parser = argparse.ArgumentParser(description="Meilisearch 日本語検索ツール")
    parser.add_argument("query", help="検索文字列")
    parser.add_argument("--limit", type=int, default=20, help="取得件数")
    args = parser.parse_args()

    results = search_meilisearch(args.query, args.limit)
    if not results:
        print("No results found.")
        return

    for i, (url, _) in enumerate(results, start=1):
        print(f"{i}\t{url}")

if __name__ == "__main__":
    main()

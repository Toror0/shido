# journals/ — 雑誌の図の決まり（1 誌または 1 出版社で 1 ファイル）

`server.py figure` と MCP の `make_journal_figure` は、ここの YAML だけを見て図を仕上げます。
雑誌の数値はコードに書いていません。雑誌を足すときは YAML を 1 つ足すだけです。

- `_出版社.yaml`（先頭が `_`）：出版社の共通の決まり。雑誌の一覧には出ません
- `雑誌.yaml`：`extends: _出版社` で共通の決まりを引き継ぎ、違う所だけ書きます（項目ごとに上書き）
- ファイル名（拡張子なし）が雑誌の id です（例 `acta_materialia`）。`aliases` の略称でも選べます

## 書く項目

| 項目 | 例 | 意味 |
|---|---|---|
| `name` / `publisher` | `Acta Materialia` / `Elsevier` | 雑誌名・出版社（必須） |
| `extends` | `_elsevier` | 引き継ぐファイル |
| `aliases` | `["Acta Mater."]` | 略称（`--journal` で使える） |
| `width_mm` | `{single: 90, one_half: 140, double: 190}` | 段の幅（mm） |
| `max_height_mm` | `248` | 最大の高さ（mm）。**決まりが無い雑誌は `null`** |
| `fonts` | `[Arial, Helvetica]` | フォントの候補（先頭から使う） |
| `text_pt` | `{body: 7, min: 6}` | 仕上がりの文字の大きさ（pt）。body＝ふつう、min＝最小 |
| `min_line_pt` | `0.5` | 最小の線の太さ（pt） |
| `dpi` | `{line: 1000, photo: 300, combination: 500}` | 線画・写真・組み合わせ（構造図＋文字はこれ） |
| `formats` | `[eps, pdf, tiff, jpeg]` | 受け付ける形式（tiff pdf eps png jpeg ps ai svg office hpgl gif） |
| `color_mode` | `RGB` | `RGB` / `CMYK` / `grayscale` |
| `panel_label` | `{style: "a", bold: true, size_pt: 8}` | パネル記号。style は `"(a)"` `"a"` `"A"` `"(A)"` `"a)"` |
| `sources` | `[{url: …, checked: 2026-11-25, note: …}]` | 調べたページと日付（必須） |
| `unverified` | `[panel_label.bold, dpi.photo]` | **確かめていない項目**（下を参照） |
| `notes` | 文 | 補足 |

## 確かめていない値は書かない

公式の投稿規定で確かめられなかった値は **埋めずに** `unverified` に項目名（`width_mm.double` のように点でつなぐ）を書きます。
使うときは安全側の既定（`crystal/core/journal_spec.py` の `SAFE_DEFAULTS`：1 段 80 mm、2 段 170 mm、高さ 225 mm、
最小の文字 7 pt、ふつうの文字 8 pt、線 0.5 pt、600 dpi、パネル記号 "(a)" 太字なし）を使い、結果の
`defaults_used` と `warnings`、`はじめに読む.html` に「未確認なので既定を使った」と出します。
値を書いたのに `unverified` にも入れるとエラーになります（どちらか 1 つ）。

読み込むときに確かめること：知らない項目・型（正の数か）・形式やパネル記号の書き方・`sources` があるか・
必須の項目（`panel_label.size_pt` 以外）が「値か unverified か」のどちらかになっているか・`extends` の行き先と循環。

## 新しい雑誌の足し方

1. 出版社の共通ファイルがあれば、雑誌のファイルは 3 行で済みます：
   ```yaml
   name: Journal of Magnetism and Magnetic Materials
   extends: _elsevier
   aliases: ["J. Magn. Magn. Mater."]
   ```
2. 出版社ごと新しいときは `_出版社.yaml` を上の表どおりに作ります（`_aps.yaml` などが見本）。
3. 確かめる：`python server.py figure --list`（一覧に出る）→ `pytest tests/test_journal_figure.py`
   （全部の雑誌で 1 段の幅の画素数を確かめるテストが自動で増えます）。

### AI に下書きさせる手順

投稿規定のページの URL を渡して、次のように頼みます（Claude・Gemini など）：

> この投稿規定のページ（URL）を読んで、`journals/README.md` の書式で `journals/<id>.yaml` の下書きを作って。
> ページに書いてある値だけを入れ、書いていない・はっきりしない値は入れずに `unverified` に項目名を足して。
> 換算した値（in → mm など）はコメントに元の数値を書いて。`sources` に URL と今日の日付。

できた下書きは、人が公式ページと見比べてから入れます（AI は値を補ってしまうことがあるため）。
`python server.py figure --list` でエラーが出なければ形は正しいです。

from mybrain.ingest.chunker import chunk_markdown, estimate_tokens

DOC = (
    """<!-- page: 1 -->
# Annual Report

## Overview
Pars Holding is an industrial group. It operates in steel and energy. The CFO is Ali Rezaei.

## Financials
| Year | Revenue | EBITDA |
| --- | --- | --- |
| 2022 | 100 | 20 |
| 2023 | 130 | 31 |

<!-- page: 2 -->
## Board
"""
    + " ".join(f"Sentence number {i} about the board of directors and governance." for i in range(80))
    + """

### Notes
Short note.
"""
)


def test_breadcrumbs_pages_and_tables():
    chunks = chunk_markdown(DOC, title="Report", max_tokens=120, overlap_tokens=30)
    assert chunks, "no chunks"
    assert all(c.breadcrumb.startswith("Report") for c in chunks)
    table_chunks = [c for c in chunks if "| 2023 | 130 | 31 |" in c.text]
    assert len(table_chunks) == 1 and "| Year |" in table_chunks[0].text, (
        "table must stay whole with its header"
    )
    # a tiny section merges into the next chunk; the heading line itself stays in the text
    assert "Financials" in table_chunks[0].breadcrumb or "## Financials" in table_chunks[0].text
    assert any(c.page == 2 for c in chunks)
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_long_paragraph_is_split_with_overlap():
    chunks = chunk_markdown(DOC, title="Report", max_tokens=120, overlap_tokens=30)
    board = [c for c in chunks if "Sentence number" in c.text]
    assert len(board) >= 3
    for c in board:
        assert c.token_count <= 120 * 1.6
    # overlap: the first sentence of a continuation chunk already appeared at the end of the previous one
    first_sentence = board[1].text.split(".")[0] + "."
    assert first_sentence in board[0].text


def test_table_longer_than_budget_is_split_by_rows_with_header_repeated():
    rows = "\n".join(f"| r{i} | value {i} | more text here {i} |" for i in range(60))
    md = "| id | value | note |\n| --- | --- | --- |\n" + rows
    chunks = chunk_markdown(md, max_tokens=80, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(c.text.startswith("| id | value | note |") for c in chunks)


def test_estimate_tokens_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") < estimate_tokens("one two three four five six")
    assert estimate_tokens("رضایی مدیر مالی") >= 3

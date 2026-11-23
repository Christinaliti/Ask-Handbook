# Why the eval failures failed

Run of 23 November 2026: `gpt-4o-mini` at temperature 0, `text-embedding-3-small`, hybrid search top 6, prompt version `9a7076107852` (see `run` in `results.json`).

**Answers 17/23 correct · Citations 17/23 · Refusals 6/6.** 8 questions failed at least one check. Each cause below was checked against the chunks the model received, not guessed.

| # | Type | Question | Why it failed |
|---|---|---|---|
| 3 | catalogue | What does part HX40-SEAL-28S-316 cost? | **The test file was incomplete.** The answer (1284.00 EUR) is right and cites p.48, which lists the same part at the same price, as does p.65; the test file only accepts p.5. |
| 4 | dutch | What criticality class is pump P-1201A, and what does it pump? | **Search didn't find the page.** OV-220 p.4 is Dutch; only the tag "P-1201A" matches as a keyword (rank 22) and it is 64th by meaning, so the model correctly said "I don't know". |
| 5 | dutch | Who has to approve postponing the four-yearly overhaul of a pump? | **Search didn't find the page.** OV-220 p.7 shares no keywords with the English question and is 64th by meaning; the model correctly said "I don't know". |
| 8 | dutch | Which spare parts must be kept on site for a criticality B pump? | **Search just missed the page, then the model answered a different question.** OV-220 p.11 is 3rd by meaning but 7th after merging, behind IOM p.30 chunks. The model listed the IOM's recommended spares, whose "Class B" is a spare-part class, not pump criticality: a wrong answer instead of a refusal. |
| 9 | two-documents | At what vibration levels do I need to act on an HX pump? | **Search didn't find the second document.** OV-220 p.8 (2.8 mm/s attention level, 1.5 mm/s step rule) is 243rd by meaning, so the answer gives only the IOM's 4.5 and 7.1 mm/s levels: correct but incomplete. |
| 10 | two-documents | What PPE do I need for an oil change, and how do I get rid of the old oil? | **The model used only one of two documents.** Search found safety register p.4 (nitrile 0.11 mm), but the model answered and cited only the IOM; the waste-code page (p.6-7, EURAL 13 02 05*) wasn't retrieved. Only the citation check fails. |
| 14 | catalogue | Can I place an emergency order for an HX-400/S volute casing? | **The model had both facts and still answered wrongly.** It got the parts list (HX40-CAS-316, column "CL" = C) and p.70 ("accepted only for class A and class B parts"), yet answered "Yes". |
| 17 | fact | Which gloves should I wear when working with SC-4 descaler? | **The model left out a detail.** "Butyl rubber, never nitrile" is right and cited, but the 0.7 mm thickness on the cited page is missing. The check is strict here, but glove thickness is part of the PPE spec. |

## In short

#8 had two causes, so it is counted under both search and the model.

- **Search, 4 failures (#4, #5, #8, #9).** All four need the Dutch OV-220. English questions rarely share keywords with Dutch text, and the embeddings rank those pages too low. Translating the question into Dutch before searching, or embedding a translation of each chunk, is the next thing to try.
- **The model, 4 failures (#8, #10, #14, #17).** Two wrong answers that sound confident (#8, #14), one answer that ignores a retrieved second source (#10), one missing detail (#17). #8 and #14 are the dangerous kind, because they look like real answers. A larger model than `gpt-4o-mini` is the obvious thing to compare.
- **The test file, 1 failure (#3).** Pages 48 and 65 should be added as accepted sources.
- **Refusals: 6/6 in this run, but not every run.** In a later run, #19 (minimum VSD speed) was answered with the catalogue's rated speeds as if they were minimum speeds. Sent five times with identical input, it was refused only 3 times. #11 also lost one of its two documents in that run. gpt-4o-mini at temperature 0 is not fully deterministic, so compare several runs before trusting a single score.

The key values for the 20 questions marked `"verified": false` are my choice, and #17 shows they can be strict. They still need checking.

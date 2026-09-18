---
name: add-paper-to-survey
description: Add one or more papers from the user's Zotero library to the Awesome-HOI-Reconstruction-and-Generation survey repo. Reads the paper, generates a structured summary using AI_SUMMARY_TEMPLATE.md, and inserts a README entry at the correct section. Skips papers whose summary already exists. Trigger when the user types `/add-paper-to-survey` followed by a Zotero collection path, paper title, itemKey, or `--queue <file>`.
---

# Add Paper to Survey

Streamline adding papers from the user's Zotero library into the
`Awesome-HOI-Reconstruction-and-Generation` survey repository.

## Repository Contract

- Survey repo root: `/Users/eric/Desktop/zgca-project/Awesome-HOI-Reconstruction-and-Generation`
- Summary template: `/Users/eric/Desktop/zgca-project/AI_SUMMARY_TEMPLATE.md`
- README: `<repo>/README.md`
- Summary destination: `<repo>/papers_summaries/<chapter_dir>/<section_dir>/<PaperName>_arXiv<year>.md`
- Filename slug: lowercase title → drop punctuation → spaces to underscores → trim. Example: `Ego-Pi: VLA Fine-Tuning for Ego-Centric Human and Robot Data` → `Ego_Pi_arXiv2026.md`

## Input Modes (parse the args after `/add-paper-to-survey`)

- **Collection path** (Chinese or English names accepted; spaces OK; quote the whole arg):
  `3DHand综述/参考综述/chapter6/Generalist Policy Learning/Structured HOI Supervision`
  → process every paper in that leaf collection that is missing a summary.
- **Single paper by title**: `--title "Ego-Pi"` (substring match).
- **Zotero itemKey**: `--key BCT9BGCC`.
- **Queue file**: `--queue PENDING_PAPERS.md` — markdown list, one entry per line:
  `- <itemKey or title> | <Zotero collection path>`.
- **No args**: look for `PENDING_PAPERS.md` in the repo root and process it.

## Workflow

### 1. Resolve Zotero source papers

Always work on a read-only copy (Zotero DB locks frequently):

```bash
cp /Users/eric/Zotero/zotero.sqlite /tmp/zotero_readonly.sqlite
```

**Preferred — Zotero MCP** (if `mcp__zotero__*` tools are loaded in this session):
- `search_collections` with the leaf name → get `collectionKey`.
- `get_collection_items` with that key → list of items.
- `get_item_details` for each item → metadata + attachments.
- `get_content` with `attachmentKey` and `mode: "complete"` → full PDF text.

**Fallback — sqlite3** (when MCP not available):

```sql
-- Find leaf collection by name (try Chinese + English aliases)
SELECT collectionID, collectionName FROM collections
WHERE collectionName IN (
  'Hand-Object Reconstruction','Hand-Object Contact and Affordance Reconstruction',
  'Hand-Held Object Reconstruction','Hand-Object Motion Reconstruction',
  'Hand-Object Grasp Generation','Hand-Object Motion Generation',
  'Hand-Object Image/Video Generation',
  'Shape Completion Priors','Shape Retrieval Priors','Spatial Geometry Priors',
  'Visual Grounding Priors','Language Reasoning Priors',
  'Image Generative Priors','Video Generative Priors',
  'Video-Based Pretraining','Structured HOI Supervision',
  'Dexterous Grasp and Motion Retargeting','Interaction-Guided Robot Manipulation',
  'Reconstruction Benchmark','Generation Benchmark','Embodied Learning Data Sources'
) AND parentCollectionID IS NOT NULL;

-- Items in that collection
SELECT i.itemID, i.key, iav.value AS title
FROM items i
JOIN itemData id ON i.itemID=id.itemID
JOIN itemDataValues iav ON id.valueID=iav.valueID
JOIN fields f ON id.fieldID=f.fieldID
JOIN collectionItems ci ON i.itemID=ci.itemID
WHERE f.fieldName='title' AND ci.collectionID=<leaf_id>;

-- Date (year) for filename
SELECT iav.value FROM items i
JOIN itemData id ON i.itemID=id.itemID
JOIN itemDataValues iav ON id.valueID=iav.valueID
JOIN fields f ON id.fieldID=f.fieldID
WHERE f.fieldName='date' AND i.itemID=<itemID>;
```

### 2. Get the PDF and extract text

```sql
SELECT ia.path FROM itemAttachments ia WHERE ia.parentItemID=<itemID>;
```

Then extract (the first 12 pages are usually enough for the summary):

```bash
pdftotext -layout -l 12 "<pdf_path>" -
```

If the paper is longer, also pull pages 13-24 with `pdftotext -layout -f 13 -l 24 ...`.
Use the `Read` tool with `pages: "1-12"` for a more accurate rendering when figures
and equations matter.

### 3. Check if summary already exists

```bash
ls "<repo>/papers_summaries/<subdir>/" | grep -i "<slug>"
```

If a matching file exists, **skip** and report. Do not overwrite.

### 4. Generate the summary

Strictly follow the 6 sections of `/Users/eric/Desktop/zgca-project/AI_SUMMARY_TEMPLATE.md`:
1. Summary (one sentence)
2. Problem and Setting
3. Core Method
4. Knowledge, Supervision, and Assumptions
5. Experiments and Findings
6. Strengths and Limitations
7. Takeaway

The "Limitations" section is part of "5. Strengths and Limitations" in the template.
Keep numbers and dataset names verbatim from the paper. If a number is unclear,
say so — do not fabricate.

### 5. Write the summary file

Path: `<repo>/papers_summaries/<subdir>/<PaperName>_arXiv<year>.md`

### 6. Insert the README bullet

Locate the section by its anchor in `README.md`, e.g. for section 5.2:

```
<a id="52-human-data-pretraining-structured-hoi-supervision"></a>
### 5.2 Human-Data Pretraining: Structured HOI Supervision
```

Insert the new entry in **reverse chronological order** within the section
(newer year first; within the same year, alphabetical by first author last name).
Use the same format as existing bullets:

```markdown
- **<PaperName>** — *<Full Title>*
  [![arXiv](https://img.shields.io/badge/arXiv-<id>-b31b1b.svg)](<arxiv_url>) [📝 Paper Summary](<summary_path>) [<optional website/github badges>]
```

If the arXiv ID is not in the paper metadata, use the placeholder:
`[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](http://arxiv.org/abs/<guess>)`.

### 7. Report

```
Processed N papers:
  ✓ <Title> → <summary_path> (section X.Y)
  ⏭ <Title> (summary already exists, skipped)
  ✗ <Title> (error: <reason>)
```

## Zotero Collection → README Section Mapping

Match by **leaf collection name**. If the leaf name doesn't appear in the table,
walk up to the parent and re-check.

| Zotero leaf name | README section | Summary subdir |
|---|---|---|
| Hand-Object Reconstruction | 1.1 | `chapter2_non_foundation/2_3_1_hand_object_recon/` |
| Hand-Object Contact and Affordance Reconstruction | 1.1 | `chapter2_non_foundation/2_3_1_hand_object_recon/` |
| Hand-Held Object Reconstruction | 1.2 | `chapter2_non_foundation/2_3_2_hand_held_object_recon/` |
| Hand-Object Motion Reconstruction | 1.3 | `chapter2_non_foundation/2_3_3_hand_object_motion_recon/` |
| Hand-Object Grasp Generation | 1.4 | `chapter2_non_foundation/2_4_1_grasp_generation/` |
| Hand-Object Motion Generation | 1.5 | `chapter2_non_foundation/2_4_2_motion_generation/` |
| Hand-Object Image/Video Generation | 1.6 | `chapter2_non_foundation/2_4_3_image_video_generation/` |
| Non-Foundation-Prior Methods for HOI Reconstruction | (parse child sub-paths) | `chapter2_non_foundation/2_3_*/` or `2_4_*/` |
| Non-Foundation-Prior Methods for HOI Generation | (parse child sub-paths) | `chapter2_non_foundation/2_4_*/` |
| Prior Source | (skip — not a paper collection) | n/a |
| Shape Retrieval Priors | 2.2 | `chapter3_3d_geometry_priors/3_3_shape_retrieval/` |
| Shape Completion Priors | 2.3 | `chapter3_3d_geometry_priors/3_2_shape_completion/` |
| Spatial Geometry Priors | 2.4 | `chapter3_3d_geometry_priors/3_4_spatial_geometry/` |
| Visual Grounding Priors | 3.2 | `chapter4_semantic_priors/4_2_visual_grounding/` |
| Language Reasoning Priors | 3.3 | `chapter4_semantic_priors/4_3_language_reasoning/` |
| Image Generative Priors | 4.3 | `chapter5_visual_motion_generative_priors/5_2_image_generative/` |
| Video Generative Priors | 4.4 | `chapter5_visual_motion_generative_priors/5_3_video_generative/` |
| Video-Based Pretraining | 5.1 | `chapter6_robot_learning/6_2_1_video_based_pretraining/` |
| Structured HOI Supervision | 5.2 | `chapter6_robot_learning/6_2_2_structured_hoi_supervision/` |
| Dexterous Grasp and Motion Retargeting | 5.3 | `chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/` |
| Interaction-Guided Robot Manipulation | 5.4 | `chapter6_robot_learning/6_3_2_interaction_guided_policy/` |
| Reconstruction Benchmark | 7.1 | `chapter7_datasets_metrics/datasets/` |
| Generation Benchmark | 7.2 | `chapter7_datasets_metrics/datasets/` |
| Embodied Learning Data Sources | 7.3 | `chapter7_datasets_metrics/datasets/` |

## Pitfalls

- The Zotero DB is **frequently locked**; always use `/tmp/zotero_readonly.sqlite`.
- The user may use either `参考文献` or `参考综述` for the same parent. Try both.
- Some Zotero leaf names contain slashes (e.g. `4D几何/可渲染重建`) — `sqlite3` returns them as-is; quote carefully.
- The MCP server and sqlite3 may return collection counts that include PDFs without metadata — filter `itemTypeID NOT IN (1, 14)` (attachment, note) if results look wrong.
- `pdftotext` may garble math — for equations and tables, use the `Read` tool on the PDF instead.
- When inserting into README, do not break the existing reverse-chronological order.

## How to extend the mapping

When the user adds a new Zotero collection that does not match any row above:
1. Pause and ask the user which README section to target (1.1–7.3).
2. Propose a new row to add to the mapping table.
3. Persist the new row by editing this skill file.

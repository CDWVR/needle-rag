# Needle — UI/UX audit and motion plan

Audited 10 Oct 2026 against the running app (`backend/main.py`, signed in as owner, 5 documents, 20 conversations) at 1440×900 and 375×812, plus a full read of `frontend/index.html`, `frontend/style.css` and `frontend/js/app.js`. One live question was asked to observe the loading and answer states.

Priorities:
- **P0**: visible in the first minute of use, or breaks trust in the answer. Fix before calling the app complete.
- **P1**: real friction in a core flow.
- **P2**: polish.

Line references are to `frontend/js/app.js` unless marked otherwise.

---

## P0 — fix before release

| # | Gap | Evidence | Fix |
|---|---|---|---|
| 1 | **Answers render raw Markdown and LaTeX.** | Live answer showed `**Global (Soft) attention**` and `\(p_t\)`, `\([p_t-D, p_t+D]\)` as literal text. `renderAnswerHtml` only escapes HTML and turns `[n]` into chips (312). | Render a safe Markdown subset (bold, italic, inline code, lists, headings, tables) after escaping, and convert `\( … \)` / `$…$` to KaTeX or at least `<code>`. Also tell the writer prompt (`backend/pipeline.py:_assemble_prompt`) which formatting the UI supports. |
| 2 | **The headline is the wrong question after a follow-up.** | Asked a follow-up: the 52 px headline stayed "What are the two types of attention…", while the new question sat in small grey "You asked" text. Same during loading (647). | Make the current question the headline. Show the conversation title once, small, above it. |
| 3 | **Earlier turns are hidden behind one-line grey links.** | A follow-up replaces the previous answer; the earlier turn becomes a 12 px muted button above the title (`turn-history`, 346). Reading a thread means clicking back and forth. | Render the conversation as a scrolling thread (question → answer → sources per turn), auto-scrolling to the newest, with the inspector following the turn in view. |
| 4 | **The evidence panel is stale while a new answer loads.** | For ~17 s after sending, the right panel showed the previous answer's trace and sources. | On submit, switch the inspector to a "working" state (see motion M1) and clear the old sources. |
| 5 | **Waiting is a 12 px grey line for ~17 s.** | Answer took 17.2 s; the only feedback was `.status-line` text. The server already sends `status` events with a `stage` field (`condense`, `search`, `retry`, `context`, `generate`, `validate`, `backend/pipeline.py:382`), but the UI uses only the message text (606). | Drive the four trace nodes and a headline-area progress row from `event.stage` (M1). Keep the answer gated: the design releases it only after the check, which is right. Make the wait legible instead. |
| 6 | **Most text is too small to read comfortably.** | Measured visible text elements under 11 px: Ask 74 of 124, Pipeline 72 of 107, Analytics 41 of 85, Knowledge 33 of 66. Many labels are 8 px DM Mono (table headers, status chips, source footers, trace labels, `style.css:363,373,397,420`). | Set a floor: 12 px for body/UI text, 11 px for uppercase mono labels, 13–14 px for table cells and descriptions. Most cards keep their look at these sizes because they're already spacious. |
| 7 | **Several colours fail WCAG AA contrast.** | Source-card footer `#91948b` on `#fbfaf5`: **2.95:1**. Placeholder `#8d9087`: **3.1:1**. `--muted #6d7168` on paper: **4.37:1** (needs 4.5). | Darken `--muted` to about `#5c6057` (≈5.3:1) and use it for footers, placeholders and table metadata. |
| 8 | **Conversations can't be renamed or deleted, and empty ones pile up.** | 4 of 20 threads are "New conversation"; 6 titles are duplicated. "+" creates a server-side conversation immediately (`#newChat`, 1600), so every click that isn't followed by a question leaves an empty thread. There's no delete or rename endpoint in `api.js`. | Create the conversation lazily on the first question. Auto-title from the first question (or a short model-written title). Add a row menu: Rename, Delete (with Undo toast). |
| 9 | **Raw validation errors reach users.** | Opening `#document/does-not-exist` shows, on the page and in a toast: *"String should match pattern '^[0-9a-f]{8}-…'"*. | Map 404/422 to plain messages ("That document no longer exists") in `errorMessage` (`api.js`). Add a proper not-found state with a way back. |
| 10 | **Dialogs don't manage focus.** | Measured: on open, focus stays on `<body>`; Tab isn't trapped; after Escape, focus doesn't return to the trigger. Affects sign-in, delete, reset, registry and sign-out. | In `openModal`: store the trigger, focus the first input (or the primary button), trap Tab inside the card, and restore focus in `closeModal`. Add `aria-describedby` for the body. |

---

## P1 — core-flow friction

### Ask

- **Citation → source navigates away.** Clicking a source card calls `openDocument` (522), which leaves the answer for the document page. That page doesn't scroll to or highlight the cited passage either. Open a side preview with the passage highlighted, plus an "Open in document" link to `#document/<id>?passage=<n>` that scrolls to and focuses it.
- **Weak sources shown as "supporting".** A 22% match appears under "Supporting sources" next to 78%/77%. The percentage isn't labelled (similarity? Jev relevance?). Label it ("Relevance 0.22") and dim passages below the keep threshold, or group them under "Related".
- **No stop, retry or edit.** While answering, the send button is just disabled. Turn it into a Stop button. After an answer, add Retry (re-run) and Edit question.
- **Copy loses citations.** "Copy answer" copies the text with bare `[1]` markers. Add "Copy with sources", which appends the numbered source list.
- **No starter prompts outside demo mode.** The empty state (`renderConversation`, 321) shows sample questions only when `state.demo`. For real workspaces, suggest 3–4 questions from indexed document headings, or the user's recent questions.
- **The scope picker is unclear.**
  - "Add context" actually *limits* the question to one document.
  - The chip then says "1 document", not the document's name (1531).
  - It's single-select with no search.

  Rename it to "Search in: All documents ▾", show the chosen document's name, and add search plus multi-select.
- **Feedback stops at a thumb.** There's no "what was wrong?" follow-up, no undo, and only a toast. A thumbs-down should open 3 quick reasons (wrong source, incomplete, should have answered) that feed Analytics.
- **Voice gives almost no visible state.** Recording is signalled only by a toast and `aria-pressed`. Show a live timer and level meter on the button, and a "Transcribing…" state in the composer.
- **Fake health bar.** The inspector's "Index health" bar is always 100% (`indexFoot`, 462). Remove it, or tie it to something real.
- **"VIEW ↗"** implies an external link but opens an in-app page. Use "→" or "Open".

### Knowledge base and documents

- **Upload is opaque.**
  - There's no drag-and-drop, and it takes one file at a time (`<input>` has no `multiple`).
  - Progress is a chain of 2.8 s toasts ("Uploading…", "Indexing…").
  - A failed row can't be retried or removed from the table: clicking it shows a toast (`openListedDocument`, 802).

  Add a dropzone on the page, multi-file queueing, and per-file progress rows driven by job polling (M6). Failed rows should offer Retry and Remove.
- **Data glitches in the table:**
  - `.txt` files show "0 B".
  - Attention Mechanism.pdf shows "—" under Updated.
  - The filter says **Synced** while the status chip says **Indexed**. Pick one word.
- **No sorting, bulk actions or inline actions.** Add sortable columns, multi-select with "Exclude / Delete", and hover actions (Exclude, Delete) on rows.
- **Collections are hard-coded** (`General / Product / Security / Research` in Settings, 1309), and a document's collection can't be changed. Either make collections editable per document, or remove the field until it does something.
- **Document page:**
  - "Open original ↗" is disabled with only a `title` tooltip explaining why. Show the reason inline.
  - The breadcrumb says "Document" instead of the file name.
  - The preview shows the first 12 passages with no way to see more.
  - URLs aren't linked.
  - Add "Load more", a search box that highlights matches, and real PDF page rendering (pdf.js) when the original is stored.

### Pipeline

- **"Refresh index" is a long job with no progress.** Only the button text changes. Show the stages (copy → embed → recall check → publish), with the recall comparison as the final gate (M10).
- **Recent runs show a clock time with no date** ("12:06:39 AM"). Use relative time plus the date for anything older than today.
- **The drift warning is developer-facing** (mentions `backend/eval/config.json`). Keep it for owners, but lead with what it means: "Questions use 15 candidates; the server default is 30."
- **"Reconcile" runs immediately.** Explain what it does before it runs (a confirm dialog with a one-line description).

### Analytics

- **The chart can't be read precisely:** no y-axis values, no legend, only a native `title` tooltip, and hairline bars at 30/90 days. Add an axis, a legend ("Asked / Grounded"), and a hover card.
- **Small samples look definitive:** "Helpful rating 100%" is from **1** rating. Show "1 rating" prominently and mute the percentage until n ≥ 10.
- **Metrics look contradictory:** Citation coverage **8.3%** next to Grounded rate **61.1%** looks alarming without a definition. Add an info tooltip, or rename it to what it measures.
- **"LIVE METRICS"** is shown, but the page never updates. Poll, or remove the badge.
- **The Knowledge-gaps "Add source" button opens a generic file picker.** Carry the gap question into the upload (e.g. "Add a source that answers: …") and re-ask it automatically after indexing.

### Settings

- **Save is easy to miss:** the button sits at the page header, far from the fields. There's no unsaved indicator and no guard when you navigate away; the draft silently survives page changes (`draftDirty`). Add a sticky save bar that appears when something changes ("2 unsaved changes · Discard · Save"), and a guard on leaving the page.
- **"Delete workspace data"** needs a typed confirmation (type the workspace name), not a single click in a dialog.
- **Retrieval numbers come with no guidance.** Show the recommended range and the default next to each field, and a "Reset to defaults" button.
- **Leftover naming:** the "Data" section's id is `billing`.

### Global

- **Toasts:** one slot, overwritten, gone after 2.8 s, no actions, and errors look the same as successes. Add stacking, an error style, actions (Undo, View, Retry), and longer display for errors.
- **The browser tab title never changes** (every page is "Needle — Knowledge Workspace"). Set `document.title` per page and per document.
- **Pages pop in after their fetch** with no loading state. Add skeletons for stat cards, tables and the inspector (M13).
- **Dates are ambiguous:** `toLocaleDateString()` gives "10/4/2026". Use "Oct 4" (or "Oct 4, 2025" for other years).
- **Ctrl/⌘ K** only focuses the thread search. Users expect a command palette (jump to page, document, conversation, or "Upload file").

### Mobile (375 px)

- The fixed 58 px rail costs 15% of the width. Use a bottom tab bar under 820 px.
- The Sources drawer opens with **no backdrop**. When closed, its shadow leaks as a grey strip along the right edge (`style.css:466`: the `box-shadow` stays while it's translated off-screen). Add a scrim, and apply the shadow only when `.open`.
- The **Threads** button shows on Knowledge, Pipeline and other pages, where it does nothing useful. Show it on Ask only.
- The question headline renders at ~45 px over 4 lines and pushes the answer below the fold. Cap it at 30–32 px on phones.
- Stat cards stack one per row, so the documents table starts about 1,000 px down. Use a 2×2 compact grid.

---

## P2 — polish

- Avatar: a lone initial ("O") is the only account control, and sign-out hides behind it. Add a small menu (name, workspace, Sign out).
- Composer: document Shift+Enter (new line) as a hint; keep the textarea from jumping by animating height.
- Thread rows: show the first answer's status (grounded / not in sources) as a small dot.
- Table and run rows: increase hit areas to at least 40 px.
- Empty states use one generic `.empty-note` style. Give the important ones (no documents, no conversations, no gaps) an illustration or the logo, plus a clear primary action.
- Consider a dark theme: the marketing and films are ink-first, while the app is paper-only.

---

## Motion plan

The app has one animation (`rise`, 12 px fade-up) and a few `.2s` transitions. A global `prefers-reduced-motion` rule exists (`style.css:519`); keep every item below behind it. Durations: 150–250 ms for UI feedback, 300–450 ms for panels, 500–700 ms only for hero moments. Use one easing family: `cubic-bezier(.2,.8,.2,1)` to enter, `cubic-bezier(.4,0,1,1)` to exit.

The **brand motif** from the launch films should carry through: the acid **fold** (dog-ear) marks a source, and the **highlighter sweep** marks a passage.

| # | Where | What moves | Why |
|---|---|---|---|
| **M1** | Ask → while answering | The inspector trace becomes a live progress strip. Each `status.stage` lights its node: `condense/search → Query/Search`, `context → Rerank`, `generate/validate → Verify`. Lines fill (scaleX 0→1, 300 ms) as stages advance. The same strip, smaller, sits under the question headline. On release, the Verify node gets the acid fold. | Turns a silent ~17 s wait into visible, truthful progress. P0 #5. |
| **M2** | Answer reveal | The answer fades up paragraph by paragraph (40 ms stagger). Citation chips pop in (scale .6→1, 220 ms, slight overshoot) after their sentence. | Makes the gated release feel like an arrival, not a jump. |
| **M3** | Citation ↔ source | Hovering `[n]` raises and outlines its source card. Clicking scrolls the card into view with a brief acid pulse, and sweeps a highlight across the matched passage (`background-size` 0→100%, 450 ms). | Shows the link between claim and evidence — the product's whole promise. |
| **M4** | Source cards | A card that's actually cited gets the dog-ear fold (the logo animation, 350 ms) as it appears. | Ties the UI to the brand and the film. |
| **M5** | Thread list | A new thread slides in at the top (height + fade, 220 ms). The active-row highlight slides between rows instead of jumping. Deleted rows collapse. | Makes list changes easy to follow. |
| **M6** | Upload | Drag-over: the dropzone border turns acid and the page scales to 1.01. Each file gets a row with a stepped progress bar: Uploading → Parsing → Chunking → Embedding → Indexed, from job polling. The finished row flashes acid-soft, then settles. | Replaces the toast chain. P1 Upload. |
| **M7** | Stat cards (Knowledge, Pipeline, Analytics) | Numbers count up on first view (600 ms, tabular figures); deltas fade in after. | Cheap and lively, and it signals fresh data. |
| **M8** | Analytics chart | Bars grow from the baseline (15 ms stagger). Switching 7/30/90 days morphs heights instead of re-rendering. Hovering shows a card with exact values. | Readability plus delight. |
| **M9** | Pipeline map | Small acid dots travel along the arrows: on the ingestion row when an upload runs, on the query row once per recent question. A node pulses when its stage last ran. | The architecture diagram becomes a live system view. |
| **M10** | Index refresh | A modal with four steps (copy → embed → recall check → publish), each with a spinner that turns into a check. The recall comparison animates as a before/after bar. | A long, risky operation needs visible stages. |
| **M11** | Modals and drawers | Backdrop fades in (180 ms). Card scales .96→1 with a fade. Mobile drawers slide in over a scrim. Exits are 30% faster than entrances. | Currently they appear instantly (measured `animation: none`). |
| **M12** | Toasts | Slide up and stack (max 3). Error toasts tint coral. A thin bar shows the remaining time and pauses on hover. | Supports the new toast actions (Undo/Retry). |
| **M13** | Loading | Skeleton shimmer for stat cards, table rows and the inspector. For the answer wait, a looping logo-fold indicator instead of text. | Removes blank-then-pop page loads. |
| **M14** | Page switches | Keep the 350 ms `rise`, but make it consistent: rail → page uses the same motion, and Document ↔ Knowledge slides left/right by direction. | Spatial continuity. |
| **M15** | Controls | Switches: knob overshoot (back ease, 200 ms). Save: the button morphs to "✓ Saved" for 1.2 s. Send turns into Stop while answering. The voice button shows a live level ring. | Feedback exactly where the action happened. |
| **M16** | First run / empty workspace | The logo builds (stems slide in, fold drops: the intro film's sting at about 1.1 s), then "Upload your first document". | Uses the new brand motion, once, where it matters. |

---

## Suggested order

1. **Quick wins (≈1 day):**
   - #1 Markdown/LaTeX rendering
   - #2 headline = current question
   - #4 clear the inspector on submit
   - #7 muted-colour token
   - #9 friendly errors
   - per-page tab titles
   - "Synced" → "Indexed"
   - mobile backdrop and shadow fix
   - Threads button on Ask only
   - date format
2. **Core flows (≈3–4 days):**
   - #3 threaded conversation
   - #5 with M1–M3 (staged progress, reveal, citation linking)
   - #8 conversation rename/delete with lazy creation
   - #10 modal focus
   - the font-size floor (#6)
3. **Knowledge and Settings (≈3 days):**
   - upload dropzone with M6
   - failed-row actions
   - sticky save bar
   - typed reset confirmation
   - analytics chart axis/legend/tooltip with M8
4. **Motion polish (≈2 days):** M4, M5, M7, M9–M16.

---

## Implementation status (branch `ux/audit-fixes`)

Done and checked in the running app (desktop 1440×900 and phone 375×812):

- **P0:** all ten.
  - #1 Markdown and math rendering (`frontend/js/markdown.js`).
  - #2–#5 threaded turns, the live stage strip, and the evidence panel that follows the turn on screen.
  - #6 font floor: text under 11 px went from 74 elements on Ask to 0, apart from the 3-letter file badges.
  - #7 contrast: grey text is now 5.6:1 on paper.
  - #8 rename/delete with Undo, and lazy conversation creation; empty threads are hidden.
  - #9 friendly errors, plus a not-found page.
  - #10 dialog focus management.
- **P1:**
  - **Ask:** citations open the passage in place, with "Open in document" jumping to and flashing the exact passage; weak and related passages are marked; Stop, Ask again, Edit, and Copy with sources; starter questions; a named scope picker with search; thumbs-down reasons and Undo; voice timer and level ring; the fake health bar is removed.
  - **Knowledge:** drag-and-drop anywhere, multi-file upload queue with per-file stages and "Try again", sortable columns, include/exclude and delete on each row, "Indexed" wording, no more "0 B".
  - **Document page:** breadcrumb shows the document name, "Open original" explains itself, "Load more", find-in-document, links work.
  - **Pipeline:** dated run log, plain-language drift note, Reconcile confirmation, refresh dialog with elapsed time.
  - **Analytics:** axis, legend, hover card, small-sample caveat, gap questions carried into the upload and re-asked.
  - **Settings:** sticky save bar, leave guard, defaults and "Reset to defaults", typed confirmation for deleting workspace data.
  - **Global:** stacked toasts with actions, per-page tab titles, skeletons, Oct 4–style dates, command menu (Ctrl/⌘ K), account menu.
  - **Mobile:** bottom tab bar, scrim behind drawers, no shadow leak, Threads only on Ask, 2×2 stats.
- **Motion:** M1–M3, M5–M9, M11–M16, plus M4 as a corner fold on cited sources. M10 is an honest indeterminate bar with elapsed time; the server doesn't report refresh stages, so none are faked.
- **Backend:**
  - `PATCH`/`DELETE /api/conversations/{id}`.
  - Feedback `reason` and `"none"` (undo).
  - Document passages paging plus `focus`.
  - Empty conversations hidden from the list.
  - Tests in `backend/tests/test_api.py` (`ConversationTests`).

Not done yet:
- PDF page rendering (pdf.js).
- Per-document collections (the list in Settings is still fixed).
- Dark theme.
- Status dots on thread rows.
- Telling the writer prompt which Markdown the UI renders.

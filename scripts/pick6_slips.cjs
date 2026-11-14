// Logs the STANDARD Pick6 slips (sizes 3/4/6/8, all markets on) using the page's own assembly code (docs/pick6_builder.js):
// one implementation, two callers. Runs under Node right after pick6_builder.py build. Appends a record to
// data/pick6_slips.jsonl only when the slip changed and only while the slate's first game has not tipped, so the last record
// per (date, size) is the slip built from the last pre-tip board. Display and logging only: nothing here is priced.
const fs = require("fs");
const path = require("path");
const { assemble } = require(path.join(__dirname, "..", "docs", "pick6_builder.js"));
const TONIGHT = path.join(__dirname, "..", "docs", "data", "pick6_tonight.json");
const LOG = path.join(__dirname, "..", "data", "pick6_slips.jsonl");

const doc = JSON.parse(fs.readFileSync(TONIGHT, "utf8"));
if (!doc.fresh || !doc.candidates || !doc.candidates.length) { console.log("[pick6-slips] nothing to log: " + (doc.why_not || "no candidates")); process.exit(0); }
if (!(doc.board_captured_at < doc.first_tip)) { console.log("[pick6-slips] the board was captured after the first tip: the slate's standard slips are frozen"); process.exit(0); }
const prior = fs.existsSync(LOG) ? fs.readFileSync(LOG, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)) : [];
let added = 0;
for (const size of doc.sizes) {
  const slip = assemble(doc.candidates, doc.markets, size);
  const legs = slip.legs.map((c) => ({ market: c.market, player: c.player, player_key: c.player_key, team: c.team, game: c.game, side: c.side, line: c.line,
    multiplier: c.multiplier, rate: c.rate, hits: c.hits, n: c.n, basis: c.basis, rank_mode: c.rank_mode, text: c.text }));
  const sig = JSON.stringify(legs.map((l) => [l.market, l.player_key, l.team, l.side, l.line, l.multiplier]));
  const last = [...prior].reverse().find((r) => r.date === doc.date && r.size === size);
  if (last && last.sig === sig) continue;
  const rec = { date: doc.date, size, built_at: new Date().toISOString().replace(/\.\d+Z$/, "Z"), board_captured_at: doc.board_captured_at, first_tip: doc.first_tip,
    complete: slip.complete, legs_built: legs.length, product: slip.product, product_label: slip.product_label, sig, legs };
  fs.appendFileSync(LOG, JSON.stringify(rec) + "\n"); prior.push(rec); added++;
  console.log(`[pick6-slips] ${doc.date} size ${size}: ${legs.length} legs${slip.complete ? "" : " (short)"}, product ${slip.product == null ? "-" : slip.product.toFixed(5)}`);
  for (const l of legs) console.log(`   ${l.market.padEnd(20)} ${l.player} (${l.team}) ${l.side}${l.line != null ? " " + l.line : ""}  ${l.multiplier}x  ${l.hits}/${l.n}`);
}
console.log(`[pick6-slips] ${added} record(s) appended`);

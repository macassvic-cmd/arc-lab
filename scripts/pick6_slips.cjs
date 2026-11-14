// Logs the STANDARD Pick6 slips using the page's own assembly code (docs/pick6_builder.js): one implementation, two callers.
// Every slate: sizes 3/4/6/8 x TWO rankings ("standard" = leader's margin / hit share, "value" = historical rate x multiplier),
// all markets on, each record tagged with its ranking so the morning grade can compare them. Runs under Node right after
// pick6_builder.py build. A record is appended to data/pick6_slips.jsonl only when the slip changed and only while the slate's
// first game has not tipped, so the last record per (date, size, ranking) is the slip built from the last pre-tip board.
// Display and logging only: the multiplier product is a Standings Points basis, never a payout.
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
for (const ranking of ["standard", "value"]) {
  for (const size of doc.sizes) {
    const table = (doc.payout && doc.payout.table) || null;
    const slip = assemble(doc.candidates, doc.markets, size, ranking, table);
    const legs = slip.legs.map((c) => ({ market: c.market, player: c.player, player_key: c.player_key, team: c.team, game: c.game, side: c.side, line: c.line,
      multiplier: c.multiplier, rate: c.rate, value: c.value, hits: c.hits, n: c.n, basis: c.basis, rank_mode: c.rank_mode, text: c.text }));
    const sig = JSON.stringify(legs.map((l) => [l.market, l.player_key, l.team, l.side, l.line, l.multiplier]));
    const last = [...prior].reverse().find((r) => r.date === doc.date && r.size === size && (r.ranking || "standard") === ranking);
    if (last && last.sig === sig && (last.payout_table || !table)) continue;   // same legs: re-log once to attach the payout table the first record lacked
    const rec = { date: doc.date, size, ranking, built_at: new Date().toISOString().replace(/\.\d+Z$/, "Z"), board_captured_at: doc.board_captured_at, first_tip: doc.first_tip,
      complete: slip.complete, legs_built: legs.length, product: slip.product, multiplier_product: slip.multiplier_product, breakeven_multiple: slip.breakeven_multiple,
      base_multiple: slip.base_multiple, payout_table: table, labels: slip.labels, sig, legs };
    fs.appendFileSync(LOG, JSON.stringify(rec) + "\n"); prior.push(rec); added++;
    console.log(`[pick6-slips] ${doc.date} ${ranking} size ${size}: ${legs.length} legs${slip.complete ? "" : " (short)"}, multiplier product ${slip.multiplier_product == null ? "-" : slip.multiplier_product.toFixed(2)}, break-even multiple ${slip.breakeven_multiple == null ? "-" : slip.breakeven_multiple.toFixed(1)}`);
  }
}
console.log(`[pick6-slips] ${added} record(s) appended`);

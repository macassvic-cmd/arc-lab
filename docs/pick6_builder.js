/* Pick6 slip assembly: the ONE implementation of the builder's rules. The page's Build button calls it in the browser and
   scripts/pick6_slips.cjs calls the same file under Node for the logged standard slips. No second copy exists anywhere.

   A slip must be able to win:
     - game-level markets (1st Point, 1st Three, 1st Dunk): at most ONE leg per game per market (only one player can score a
       game's first point: two legs on the same game + market can never both hit)
     - team-level markets (Team 1st FG, Team 1st Three): at most one leg per team per market
     - at most 1 leg per player, at most 2 legs per game
   Fill order: the selected markets take turns, in board order; each turn a market offers its best remaining candidate (the
   file's own ranking: firsts by the leader's margin over the unit's #2, window legs by hit-rate share against tonight's
   line). A slip that cannot reach the requested size is returned short, and says so. */
(function (root) {
  "use strict";
  var GAME_LEVEL = { "1st Point": 1, "1st Three": 1, "1st Dunk": 1 };
  var TEAM_LEVEL = { "Team 1st FG": 1, "Team 1st Three": 1 };

  function assemble(candidates, markets, size) {
    var queues = {}, legs = [], players = {}, perGame = {}, gameMarket = {}, teamMarket = {};
    markets.forEach(function (m) {
      queues[m] = candidates.filter(function (c) { return c.market === m; }).sort(function (a, b) { return a.order - b.order; });
    });
    function allowed(c) {
      if (players[c.player_key + "|" + c.team]) return false;
      if ((perGame[c.game] || 0) >= 2) return false;
      if (GAME_LEVEL[c.market] && gameMarket[c.market + "|" + c.game]) return false;
      if (TEAM_LEVEL[c.market] && teamMarket[c.market + "|" + c.team]) return false;
      return true;
    }
    var progress = true;
    while (legs.length < size && progress) {
      progress = false;
      for (var i = 0; i < markets.length && legs.length < size; i++) {
        var q = queues[markets[i]];
        while (q.length) {
          var c = q.shift();
          if (!allowed(c)) continue;
          legs.push(c); players[c.player_key + "|" + c.team] = 1; perGame[c.game] = (perGame[c.game] || 0) + 1;
          if (GAME_LEVEL[c.market]) gameMarket[c.market + "|" + c.game] = 1;
          if (TEAM_LEVEL[c.market]) teamMarket[c.market + "|" + c.team] = 1;
          progress = true;
          break;
        }
      }
    }
    var product = legs.reduce(function (p, c) { return p * c.rate; }, 1);
    return { size: size, legs: legs, complete: legs.length === size, product: legs.length ? product : null,
             product_label: "historical rate, not a prediction" };
  }

  function legText(c) {
    return c.player + " (" + c.team + ") — " + c.market + (c.line != null ? " " + c.side + " " + c.line : " " + c.side) + " — " + c.game + " — " + c.multiplier + "x — " + c.text;
  }

  var api = { assemble: assemble, legText: legText };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Pick6 = api;
})(typeof self !== "undefined" ? self : this);

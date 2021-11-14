# Cute Lil' Plasma bot

import rgkit.rg as rg
from rgkit import rg


class Robot:
    def act(self, game):
        r = 99
        g = 0, 0
        a = self.location
        b = game['robots']
        j = rg.toward
        x = 'attack'
        e = 'move'
        s = 'spawn'
        y = rg.locs_around
        h = rg.wdist
        k = [e, j(a, rg.CENTER_POINT)]
        l = 0
        t = []
        if not game['turn'] % 10 and s in rg.loc_types(a): return k
        m = y(a, filter_out=('obstacle', s))
        for f, z in b.items():
            if z.player_id != self.player_id:
                if h(f, a) < r: r = h(f, a);g = f
                t.append(f)
                if h(f, a) == 1: l += 1
        c = j(a, g)
        u = h(a, g)
        v = u == 1
        if b[a].hp < l * 10 and v:
            n = tuple(map(lambda o, p: o - p, a, tuple(map(lambda o, p: o - p, j(a, g), a))))
            if n in m and n not in b: return [e, n]
        if v and b[a].hp > 10 * l or u == 2 and b[a].hp < 16:
            if c in t and b[c].hp < 6 and self.hp > 5: return [e, c]
            return [x, c]
        if c in m and c not in b: return [e, c]
        if k[1] in b:
            for w in m:
                if w not in b: return [e, w]
        return k



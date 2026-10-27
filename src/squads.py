"""Public squad lists of the two clubs (2024-25 season) for commentary name spotting (round 11, idea B1). Public
knowledge, not competition data. Each alias is a lower-case, accent-free token sequence; aliases that could name
a player of either club, a coach or a common word are left out on purpose (e.g. "inigo", "marc", "unai", "pena").

HOME = FC Barcelona (BAR), AWAY = Athletic Club (ATH).
"""
import re
import unicodedata

HOME_PLAYERS = {
    "ter stegen": ["ter stegen", "terstegen", "ter-stegen"],
    "inaki pena": ["inaki pena"],
    "astralaga": ["astralaga"],
    "kounde": ["kounde", "jules kounde"],
    "cubarsi": ["cubarsi", "pau cubarsi"],
    "inigo martinez": ["inigo martinez"],
    "balde": ["balde", "alejandro balde", "alex balde"],
    "araujo": ["araujo"],
    "christensen": ["christensen"],
    "eric garcia": ["eric garcia", "eric"],
    "hector fort": ["hector fort"],
    "gerard martin": ["gerard martin"],
    "sergi dominguez": ["sergi dominguez"],
    "pedri": ["pedri"],
    "de jong": ["de jong", "frenkie"],
    "gavi": ["gavi"],
    "fermin": ["fermin"],
    "casado": ["casado", "marc casado"],
    "bernal": ["bernal", "marc bernal"],
    "olmo": ["olmo", "dani olmo"],
    "pablo torre": ["pablo torre"],
    "lewandowski": ["robert lewandowski", "lewandowski", "lewandoski", "levandoski", "lewy"],
    "lamine yamal": ["lamine yamal", "lamine", "yamal"],
    "raphinha": ["raphinha", "rafinha"],
    "ferran torres": ["ferran torres", "ferran"],
    "ansu fati": ["ansu fati", "ansu", "fati"],
    "pau victor": ["pau victor"],
    "guiu": ["guiu"],
}
AWAY_PLAYERS = {
    "unai simon": ["unai simon"],
    "agirrezabala": ["julen agirrezabala", "agirrezabala", "aguirrezabala", "julen"],
    "padilla": ["padilla"],
    "de marcos": ["de marcos", "oscar de marcos"],
    "gorosabel": ["gorosabel"],
    "vivian": ["vivian", "dani vivian"],
    "paredes": ["paredes", "aitor paredes"],
    "yeray": ["yeray"],
    "yuri": ["yuri berchiche", "yuri", "berchiche"],
    "lekue": ["lekue"],
    "boiro": ["boiro", "adama"],
    "galarreta": ["galarreta", "ruiz de galarreta"],
    "vesga": ["vesga", "mikel vesga"],
    "prados": ["benat prados", "prados", "benat"],
    "jauregizar": ["mikel jauregizar", "jauregizar", "jauregi"],
    "sancet": ["oihan sancet", "sancet", "oihan"],
    "unai gomez": ["unai gomez"],
    "herrera": ["herrera", "ander herrera"],
    "berenguer": ["berenguer", "alex berenguer"],
    "inaki williams": ["inaki williams", "inaki"],
    "nico williams": ["nico williams", "nico"],
    "williams": ["williams"],
    "guruzeta": ["guruzeta", "gorka guruzeta"],
    "djalo": ["djalo", "yalo", "alvaro djalo"],
    "adu ares": ["adu ares"],
    "serrano": ["serrano"],
}
# team names: weaker evidence ("el Barca sale" vs "contra el Barca"), kept as a separate channel
HOME_TEAM = ["barca", "barcelona", "azulgrana", "azulgranas", "blaugrana", "cule", "cules"]
AWAY_TEAM = ["athletic", "bilbao", "rojiblanco", "rojiblancos", "leones"]


def norm(word):
    w = unicodedata.normalize("NFKD", word.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return re.sub(r"[^a-z\-]", "", w)


def alias_table():
    """{token tuple: (team 0 home / 1 away, kind 'player' | 'team', canonical name)}"""
    out = {}
    for team, players in ((0, HOME_PLAYERS), (1, AWAY_PLAYERS)):
        for canon, aliases in players.items():
            for a in aliases:
                out[tuple(a.split())] = (team, "player", canon)
    for team, names in ((0, HOME_TEAM), (1, AWAY_TEAM)):
        for a in names:
            out[(a,)] = (team, "team", a)
    return out


def prompt_names():
    """Spelled names for the ASR prompt (helps Whisper spell Basque / Catalan surnames)."""
    return ("Ter Stegen, Koundé, Cubarsí, Íñigo Martínez, Balde, Casadó, Bernal, Pedri, Lamine Yamal, Raphinha, "
            "Lewandowski, Ferran Torres, Fermín, Eric García, Ansu Fati, Pau Víctor, Gerard Martín, Héctor Fort, "
            "Agirrezabala, De Marcos, Gorosabel, Vivian, Paredes, Yeray, Yuri Berchiche, Lekue, Boiro, Galarreta, "
            "Vesga, Prados, Jauregizar, Sancet, Unai Gómez, Herrera, Berenguer, Iñaki Williams, Nico Williams, "
            "Guruzeta, Djaló, Barça, Athletic.")


def spot(words):
    """words: list of (start_s, end_s, text). Returns mentions [(start_s, end_s, team, kind, canon)], longest
    alias first, no overlapping matches."""
    table = alias_table()
    maxlen = max(len(k) for k in table)
    toks = [norm(w) for _, _, w in words]
    out, i = [], 0
    while i < len(toks):
        hit = None
        for L in range(maxlen, 0, -1):
            key = tuple(toks[i:i + L])
            if len(key) == L and key in table:
                hit = (L, table[key])
                break
        if hit:
            L, (team, kind, canon) = hit
            out.append((words[i][0], words[i + L - 1][1], team, kind, canon))
            i += L
        else:
            i += 1
    return out

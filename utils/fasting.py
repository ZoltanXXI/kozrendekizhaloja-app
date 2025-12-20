FASTING_RECIPE_TITLES = {
    "Káposzta ikrával", "Alma-lév", "Mondola-perec", "Koldus-lév", "Ég-lév",
    "Zsákvászonnal", "Gutta-lév", "Szíjalt rák", "Lengyel cibre", "Körtvély főve",
    "Saláta", "Torzsa-saláta", "Ugorka-saláta", "Miskuláncia-saláta", "Mondola-lév",
    "Bot-lév", "Kendermag-cibre", "Ikrát főzni", "Nyers káposzta-saláta", "Borsóleves",
    "Párolt rák", "Korpa-cibre", "Borsót főzni", "Ugorkát télre sózni", "Fenyőgombát főzni",
    "Kínzott kása", "Lencseleves", "Hal rizskásával", "Olaj-spék", "Cicer",
    "Sült hal", "Lémonyával", "Törött lével hal", "Csukát csuka-lével", "Olajos domika",
    "Kozák-lével", "Zöld lével", "Borsos szilva", "Ecetes cibre", "Hal fekete lével",
    "Zuppon-lév", "Tiszta borssal", "Bors-porral", "Vizát viza-lével", "Szömörcsök-gomba",
    "Borított lév", "Kása olajjal", "Lencse olajjal", "Borsó laskával", "Káposztás béles",
    "Hagyma rántva", "Káposzta-lév cibre", "Lönye", "Lása", "Sós víz",
    "Seres kenyér", "Olajos lév", "Viza ikra", "Új káposzta"
}

def is_fasting_title(title: str) -> bool:
    if not isinstance(title, str):
        return False
    t = title.strip()
    return t in FASTING_RECIPE_TITLES

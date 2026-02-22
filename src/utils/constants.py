"""Constants for the Passos Magicos ML pipeline.

Column normalization map, feature lists, and shared constants used
across all pipeline scripts. Column names with accented characters
match the actual UTF-8 CSV headers exactly.
"""

RANDOM_STATE: int = 42

TARGET_COL: str = "defasagem"
BINARY_TARGET_COL: str = "target"
YEAR_COL: str = "year"

COLUMN_MAP: dict[int, dict[str, str]] = {
    2022: {
        "RA": "ra",
        "Fase": "fase",
        "INDE 22": "inde",
        "Pedra 22": "pedra",
        "Turma": "turma",
        "Nome": "nome",
        "Idade 22": "idade",
        "G\u00eanero": "genero",
        "Ano ingresso": "ano_ingresso",
        "Institui\u00e7\u00e3o de ensino": "instituicao",
        "IAA": "iaa",
        "IEG": "ieg",
        "IPS": "ips",
        "IPV": "ipv",
        "IDA": "ida",
        "IAN": "ian",
        "Matem": "mat",
        "Portug": "por",
        "Ingl\u00eas": "ing",
        "Defas": "defasagem",
        "Fase ideal": "fase_ideal",
        "Rec Psicologia": "rec_psicologia",
    },
    2023: {
        "RA": "ra",
        "Fase": "fase",
        "INDE 2023": "inde",
        "Pedra 2023": "pedra",
        "Turma": "turma",
        "Nome Anonimizado": "nome",
        "Idade": "idade",
        "G\u00eanero": "genero",
        "Ano ingresso": "ano_ingresso",
        "Institui\u00e7\u00e3o de ensino": "instituicao",
        "IAA": "iaa",
        "IEG": "ieg",
        "IPS": "ips",
        "IPP": "ipp",
        "IPV": "ipv",
        "IDA": "ida",
        "IAN": "ian",
        "Mat": "mat",
        "Por": "por",
        "Ing": "ing",
        "Defasagem": "defasagem",
        "Fase Ideal": "fase_ideal",
        "Rec Psicologia": "rec_psicologia",
    },
    2024: {
        "RA": "ra",
        "Fase": "fase",
        "INDE 2024": "inde",
        "Pedra 2024": "pedra",
        "Turma": "turma",
        "Nome Anonimizado": "nome",
        "Idade": "idade",
        "G\u00eanero": "genero",
        "Ano ingresso": "ano_ingresso",
        "Institui\u00e7\u00e3o de ensino": "instituicao",
        "IAA": "iaa",
        "IEG": "ieg",
        "IPS": "ips",
        "IPP": "ipp",
        "IPV": "ipv",
        "IDA": "ida",
        "IAN": "ian",
        "Mat": "mat",
        "Por": "por",
        "Ing": "ing",
        "Defasagem": "defasagem",
        "Fase Ideal": "fase_ideal",
        "Rec Psicologia": "rec_psicologia",
        "Escola": "escola",
        "Ativo/ Inativo": "ativo",
    },
}

DUPLICATE_COLS: dict[int, list[str]] = {
    2023: ["Destaque IPV"],
    2024: ["Ativo/ Inativo"],
}

EVALUATOR_COLS_PATTERN: list[str] = [
    "Cg", "Cf", "Ct",
    "N\u00ba Av",
    "Avaliador1", "Rec Av1",
    "Avaliador2", "Rec Av2",
    "Avaliador3", "Rec Av3",
    "Avaliador4", "Rec Av4",
    "Avaliador5",
    "Avaliador6",
    "Indicado", "Atingiu PV",
    "Destaque IEG", "Destaque IDA", "Destaque IPV",
    "Pedra 20", "Pedra 21", "Pedra 23",
    "INDE 23",
    "Data de Nasc",
    "Ano nasc",
]

UNIFIED_COLUMNS: list[str] = [
    "ra", "fase", "inde", "pedra", "turma", "nome",
    "idade", "genero", "ano_ingresso", "instituicao",
    "iaa", "ieg", "ips", "ipp", "ipv", "ida", "ian",
    "mat", "por", "ing", "defasagem", "fase_ideal",
    "rec_psicologia", "year",
]

NUMERIC_FEATURES: list[str] = [
    "inde", "iaa", "ieg", "ips", "ipp", "ipv",
    "ida", "ian", "mat", "por", "ing", "fase", "idade",
]

CATEGORICAL_FEATURES: list[str] = ["genero", "instituicao", "pedra"]

COLUMNS_TO_DROP: list[str] = [
    "nome", "ra", "turma", "rec_psicologia", "fase_ideal",
    "ano_ingresso", "ativo", "escola", "defasagem", "year",
]

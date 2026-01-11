import base64
import json
from dataclasses import dataclass

import pandas as pd
import requests
import streamlit as st

# =========================
# Config UI
# =========================
st.set_page_config(page_title="Transferências Internacionais", layout="wide")

STATUS_OPTIONS = ["", "Ok", "Não-Ok"]  # "" = pendente
STATUS_LABELS = {"": "(vazio) Pendente", "Ok": "Ok", "Não-Ok": "Não-Ok"}


def normalize_status(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s if s in STATUS_OPTIONS else ""


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia colunas duplicadas adicionando sufixos __2, __3...
    Garante nomes únicos (necessário para st.data_editor).
    """
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__{seen[c]}")
    out = df.copy()
    out.columns = new_cols
    return out


def first_col_as_series(df: pd.DataFrame, colname: str) -> pd.Series:
    """
    Se df[colname] retornar DataFrame (colunas duplicadas), pega a primeira coluna como Series.
    Caso contrário, retorna a Series.
    """
    obj = df[colname]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


@dataclass
class GitHubCfg:
    token: str
    repo: str
    branch: str
    csv_path: str


def get_cfg_from_secrets() -> GitHubCfg:
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
        branch = st.secrets.get("BRANCH", "main")
        csv_path = st.secrets.get("CSV_PATH", "Transferencias_Internacionais_ATUALIZADO.csv")
        return GitHubCfg(token=token, repo=repo, branch=branch, csv_path=csv_path)
    except Exception:
        st.error(
            "Secrets não configurados. Configure em Streamlit Cloud → Settings → Secrets:\n"
            "GITHUB_TOKEN, GITHUB_REPO, BRANCH (opcional), CSV_PATH."
        )
        st.stop()


def gh_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "streamlit-app",
    }


def gh_get_file(cfg: GitHubCfg) -> tuple[str, str]:
    url = f"https://api.github.com/repos/{cfg.repo}/contents/{cfg.csv_path}"
    r = requests.get(url, headers=gh_headers(cfg.token), params={"ref": cfg.branch}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(
            f"GitHub GET falhou ({r.status_code}). Verifique CSV_PATH/BRANCH.\nResposta: {r.text}"
        )

    data = r.json()
    if isinstance(data, list):
        raise RuntimeError(f"CSV_PATH aponta para uma pasta, não um arquivo: {cfg.csv_path}")

    sha = data.get("sha", "")
    download_url = data.get("download_url")

    if download_url:
        rr = requests.get(download_url, timeout=60)
        if rr.status_code != 200:
            raise RuntimeError(f"Falha ao baixar CSV pelo download_url ({rr.status_code}).")
        content = rr.content.decode("utf-8-sig", errors="replace")
    else:
        content_b64 = data.get("content", "")
        if not content_b64:
            raise RuntimeError("GitHub não retornou content nem download_url.")
        content = base64.b64decode(content_b64).decode("utf-8-sig", errors="replace")

    if not content.strip():
        raise RuntimeError("Conteúdo do CSV no GitHub está vazio.")
    return content, sha


def gh_put_file(cfg: GitHubCfg, new_text: str, sha: str, message: str) -> None:
    url = f"https://api.github.com/repos/{cfg.repo}/contents/{cfg.csv_path}"
    payload = {
        "message": message,
        "content": base64.b64encode(new_text.encode("utf-8-sig")).decode("utf-8"),
        "sha": sha,
        "branch": cfg.branch,
    }
    r = requests.put(url, headers=gh_headers(cfg.token), data=json.dumps(payload), timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT falhou ({r.status_code}): {r.text}")


@st.cache_data(show_spinner=False)
def load_df_from_github(cfg: GitHubCfg) -> tuple[pd.DataFrame, str, str]:
    raw_text, sha = gh_get_file(cfg)

    df = pd.read_csv(pd.io.common.StringIO(raw_text))
    df = dedupe_columns(df)

    required = ["NOME DO ATLETA", "DE", "PARA", "PAÍS", "DATA", "STATUS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV não contém colunas esperadas: {missing}")

    df["DATA_DT"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
    df["ANO"] = df["DATA_DT"].dt.year
    df["STATUS"] = df["STATUS"].apply(normalize_status)

    df = df.reset_index(drop=False).rename(columns={"index": "ROW_ID"})
    df = dedupe_columns(df)

    return df, sha, raw_text


def dataframe_to_csv_text(df_full: pd.DataFrame) -> str:
    df_out = df_full.drop(columns=["DATA_DT", "ANO"], errors="ignore").copy()
    df_out = dedupe_columns(df_out)

    # Mantém apenas a primeira ocorrência de cada "base" (antes de __N)
    cols_keep = []
    seen_base = set()
    for c in df_out.columns:
        base = c.split("__")[0]
        if base in seen_base:
            continue
        seen_base.add(base)
        cols_keep.append(c)
    df_out = df_out[cols_keep].copy()

    return df_out.to_csv(index=False, encoding="utf-8-sig")


def dataframe_to_csv_bytes(df_full: pd.DataFrame) -> bytes:
    return dataframe_to_csv_text(df_full).encode("utf-8-sig")


def apply_status_updates(df_full: pd.DataFrame, edited_rows: pd.DataFrame) -> pd.DataFrame:
    df_new = df_full.copy()

    upd = edited_rows[["ROW_ID", "STATUS"]].copy()
    upd["STATUS"] = upd["STATUS"].apply(normalize_status)
    upd_map = dict(zip(upd["ROW_ID"], upd["STATUS"]))

    rid = first_col_as_series(df_new, "ROW_ID").tolist()
    old = df_new["STATUS"].tolist()
    df_new["STATUS"] = [upd_map.get(r, s) for r, s in zip(rid, old)]
    return df_new


def build_resumo_por_atleta(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame(columns=["NOME DO ATLETA", "ocorrencias", "ok", "nao_ok", "pendente", "tem_pendencia"])
    g = df_in.groupby("NOME DO ATLETA", dropna=False)
    ocorr = g.size().rename("ocorrencias")
    stats = g["STATUS"].agg(
        ok=lambda s: (s == "Ok").sum(),
        nao_ok=lambda s: (s == "Não-Ok").sum(),
        pendente=lambda s: (s == "").sum(),
    )
    resumo = pd.concat([ocorr, stats], axis=1).reset_index()
    resumo["tem_pendencia"] = resumo["pendente"] > 0
    return resumo


def apply_filters_for_list(df_in: pd.DataFrame, ano_sel, status_sel, busca) -> pd.DataFrame:
    df_f = df_in.copy()
    if ano_sel != "Todos":
        df_f = df_f[df_f["ANO"] == int(ano_sel)]
    if status_sel:
        df_f = df_f[df_f["STATUS"].isin(status_sel)]
    else:
        df_f = df_f.iloc[0:0]
    if busca:
        df_f = df_f[df_f["NOME DO ATLETA"].str.contains(busca, case=False, na=False)]
    return df_f


def save_to_github(cfg: GitHubCfg, current_df: pd.DataFrame, context_msg: str = "") -> None:
    new_text = dataframe_to_csv_text(current_df)

    if new_text.strip() == st.session_state["_gh_raw"].strip():
        st.info("Nenhuma alteração detectada para salvar.")
        return

    commit_msg = "Atualiza STATUS via app Streamlit"
    if context_msg:
        commit_msg += f" - {context_msg}"

    try:
        gh_put_file(cfg, new_text=new_text, sha=st.session_state["_gh_sha"], message=commit_msg)
        st.success("Salvo no GitHub com commit ✅")
    except Exception:
        st.warning("Arquivo mudou no GitHub. Tentando novamente com SHA mais recente...")
        latest_raw, latest_sha = gh_get_file(cfg)
        gh_put_file(cfg, new_text=new_text, sha=latest_sha, message=commit_msg)
        st.success("Salvo no GitHub com commit ✅ (retry)")

    load_df_from_github.clear()
    df_base2, sha2, raw2 = load_df_from_github(cfg)
    st.session_state["df_work"] = df_base2.copy()
    st.session_state["_gh_sha"] = sha2
    st.session_state["_gh_raw"] = raw2
    st.session_state["view"] = "lista"
    st.rerun()


# =========================
# Session state init
# =========================
for k, v in {
    "view": "lista",
    "athlete": "",
    "df_work": None,
    "_gh_sha": "",
    "_gh_raw": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# Load GitHub
# =========================
cfg = get_cfg_from_secrets()

try:
    df_base, sha_base, raw_base = load_df_from_github(cfg)
except Exception as e:
    st.error(f"Erro ao carregar do GitHub: {e}")
    st.stop()

if st.session_state["df_work"] is None:
    st.session_state["df_work"] = df_base.copy()
    st.session_state["_gh_sha"] = sha_base
    st.session_state["_gh_raw"] = raw_base

df = st.session_state["df_work"]

# =========================
# Sidebar
# =========================
st.sidebar.header("Fonte")
st.sidebar.write(f"Repo: `{cfg.repo}`")
st.sidebar.write(f"Branch: `{cfg.branch}`")
st.sidebar.write(f"CSV_PATH: `{cfg.csv_path}`")

st.sidebar.divider()
st.sidebar.header("Filtros")

anos_disponiveis = sorted([int(a) for a in df["ANO"].dropna().unique()])
ano_sel = st.sidebar.selectbox("Ano (DATA)", options=["Todos"] + anos_disponiveis, index=0)

status_sel = st.sidebar.multiselect(
    "STATUS",
    options=STATUS_OPTIONS,
    default=STATUS_OPTIONS,
    format_func=lambda s: STATUS_LABELS.get(s, str(s)),
)

busca = st.sidebar.text_input("Buscar atleta (contém)", value="").strip()

st.sidebar.divider()
cA, cB = st.sidebar.columns(2)
with cA:
    if st.button("Recarregar do GitHub", use_container_width=True):
        load_df_from_github.clear()
        df_base2, sha2, raw2 = load_df_from_github(cfg)
        st.session_state["df_work"] = df_base2.copy()
        st.session_state["_gh_sha"] = sha2
        st.session_state["_gh_raw"] = raw2
        st.session_state["view"] = "lista"
        st.session_state["athlete"] = ""
        st.rerun()
with cB:
    if st.button("Ir p/ Lista", use_container_width=True):
        st.session_state["view"] = "lista"
        st.rerun()


# =========================
# Views
# =========================
def view_lista():
    st.title("Transferências Internacionais — Atletas")

    df_f = apply_filters_for_list(df, ano_sel, status_sel, busca)
    resumo = build_resumo_por_atleta(df_f)

    top1, top2, top3 = st.columns([1, 1, 2], vertical_alignment="center")
    with top1:
        if st.button("💾 Salvar no GitHub", type="primary", use_container_width=True):
            save_to_github(cfg, df, context_msg="lista")
    with top2:
        st.download_button(
            "⬇️ Baixar CSV (opcional)",
            data=dataframe_to_csv_bytes(df),
            file_name="Transferencias_Internacionais_ATUALIZADO.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with top3:
        st.caption("Salvar no GitHub grava definitivamente o CSV no repositório (commit).")

    if resumo.empty:
        st.info("Nenhum atleta encontrado com os filtros atuais.")
        return

    resumo = resumo.sort_values(["tem_pendencia", "ocorrencias", "NOME DO ATLETA"], ascending=[False, False, True])
    sem_pend = resumo[~resumo["tem_pendencia"]].copy()
    com_pend = resumo[resumo["tem_pendencia"]].copy()

    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        st.subheader(f"✅ Atletas sem pendências ({len(sem_pend)})")
        st.dataframe(sem_pend.drop(columns=["tem_pendencia"]), use_container_width=True, hide_index=True)

        atletas_sem = sem_pend["NOME DO ATLETA"].tolist()
        if atletas_sem:
            atleta_sel = st.selectbox("Abrir ficha (sem pendências)", atletas_sem, key="sel_sem")
            if st.button("Abrir", use_container_width=True, key="btn_sem"):
                st.session_state["athlete"] = atleta_sel
                st.session_state["view"] = "ficha"
                st.rerun()

    with col2:
        st.subheader(f"⚠️ Atletas com pendências ({len(com_pend)})")
        st.dataframe(com_pend.drop(columns=["tem_pendencia"]), use_container_width=True, hide_index=True)

        atletas_com = com_pend["NOME DO ATLETA"].tolist()
        if atletas_com:
            atleta_sel2 = st.selectbox("Abrir ficha (com pendências)", atletas_com, key="sel_com")
            if st.button("Abrir", type="primary", use_container_width=True, key="btn_com"):
                st.session_state["athlete"] = atleta_sel2
                st.session_state["view"] = "ficha"
                st.rerun()


def view_ficha():
    atleta = st.session_state["athlete"]
    if not atleta:
        st.session_state["view"] = "lista"
        st.rerun()

    st.title(f"Ficha do atleta: {atleta}")

    bar1, bar2, bar3 = st.columns([1, 1, 2], vertical_alignment="center")
    with bar1:
        if st.button("← Voltar", use_container_width=True):
            st.session_state["view"] = "lista"
            st.rerun()
    with bar2:
        if st.button("💾 Salvar no GitHub", type="primary", use_container_width=True):
            save_to_github(cfg, df, context_msg=f"atleta {atleta}")
    with bar3:
        st.caption("Ocorrências ordenadas: data mais antiga → mais recente.")

    base = df[df["NOME DO ATLETA"] == atleta].copy()

    # Garante que ROW_ID é 1D (Series) e cria uma coluna "ROW_ID" única para exibição
    rid_series = first_col_as_series(base, "ROW_ID")
    base = base.copy()
    base["ROW_ID"] = rid_series

    df_a = base[["ROW_ID", "DE", "PARA", "PAÍS", "DATA", "STATUS"]].copy()

    # Ordenação robusta
    df_a["__DATA_SORT__"] = pd.to_datetime(df_a["DATA"], dayfirst=True, errors="coerce")
    df_a["__RID__"] = pd.to_numeric(df_a["ROW_ID"], errors="coerce")
    df_a = df_a.sort_values(["__DATA_SORT__", "__RID__"], ascending=[True, True], na_position="last")

    # O data_editor NÃO aceita colunas duplicadas
    df_show = df_a[["ROW_ID", "DE", "PARA", "PAÍS", "DATA", "STATUS"]].copy()
    df_show = dedupe_columns(df_show)

    edited = st.data_editor(
        df_show,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "ROW_ID": st.column_config.NumberColumn("ID", disabled=True),
            "DE": st.column_config.TextColumn("DE", disabled=True),
            "PARA": st.column_config.TextColumn("PARA", disabled=True),
            "PAÍS": st.column_config.TextColumn("PAÍS", disabled=True),
            "DATA": st.column_config.TextColumn("DATA", disabled=True),
            "STATUS": st.column_config.SelectboxColumn("STATUS", options=STATUS_OPTIONS),
        },
        disabled=["ROW_ID", "DE", "PARA", "PAÍS", "DATA"],
        key="editor_ficha",
    )

    st.divider()
    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    with c1:
        if st.button("Aplicar alterações (sessão)", use_container_width=True):
            st.session_state["df_work"] = apply_status_updates(df, edited)
            st.success("Alterações aplicadas (em memória). Agora você pode salvar no GitHub.")
            st.rerun()
    with c2:
        st.download_button(
            "⬇️ Baixar CSV (opcional)",
            data=dataframe_to_csv_bytes(df),
            file_name="Transferencias_Internacionais_ATUALIZADO.csv",
            mime="text/csv",
            use_container_width=True,
        )


# Router
if st.session_state["view"] == "lista":
    view_lista()
else:
    view_ficha()


import base64
import json
import os
from dataclasses import dataclass

import pandas as pd
import requests
import streamlit as st

# =========================
# Config UI
# =========================
st.set_page_config(page_title="Transferências Internacionais", layout="wide")

STATUS_OPTIONS = ["", "Ok", "Não-Ok"]  # "" = pendente / não marcado
STATUS_LABELS = {"": "(vazio) Pendente", "Ok": "Ok", "Não-Ok": "Não-Ok"}


def normalize_status(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s if s in STATUS_OPTIONS else ""


@dataclass
class GitHubCfg:
    token: str
    repo: str      # "user/repo"
    branch: str    # "main"
    csv_path: str  # "data/arquivo.csv"


def get_cfg_from_secrets() -> GitHubCfg:
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
        branch = st.secrets.get("BRANCH", "main")
        csv_path = st.secrets.get("CSV_PATH", "Transferencias_Internacionais_ATUALIZADO.csv")
        return GitHubCfg(token=token, repo=repo, branch=branch, csv_path=csv_path)
    except Exception:
        st.error(
            "Secrets não configurados. Configure em Streamlit Cloud:\n"
            "GITHUB_TOKEN, GITHUB_REPO, BRANCH (opcional), CSV_PATH."
        )
        st.stop()


def gh_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }


def gh_get_file(cfg: GitHubCfg) -> tuple[str, str]:
    """
    Retorna (conteudo_texto, sha)
    """
    url = f"https://api.github.com/repos/{cfg.repo}/contents/{cfg.csv_path}"
    r = requests.get(url, headers=gh_headers(cfg.token), params={"ref": cfg.branch}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub GET falhou ({r.status_code}): {r.text}")

    data = r.json()
    sha = data["sha"]
    content_b64 = data["content"]
    content = base64.b64decode(content_b64).decode("utf-8-sig", errors="replace")
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
    """
    Retorna (df, sha, raw_text)
    """
    raw_text, sha = gh_get_file(cfg)
    df = pd.read_csv(pd.io.common.StringIO(raw_text))

    required = ["NOME DO ATLETA", "DE", "PARA", "PAÍS", "DATA", "STATUS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV não contém colunas esperadas: {missing}")

    df["DATA_DT"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
    df["ANO"] = df["DATA_DT"].dt.year
    df["STATUS"] = df["STATUS"].apply(normalize_status)

    df = df.reset_index(drop=False).rename(columns={"index": "ROW_ID"})
    return df, sha, raw_text


def dataframe_to_csv_text(df_full: pd.DataFrame) -> str:
    df_out = df_full.drop(columns=["DATA_DT", "ANO"], errors="ignore").copy()
    # utf-8-sig ajuda no Excel
    return df_out.to_csv(index=False, encoding="utf-8-sig")


def dataframe_to_csv_bytes(df_full: pd.DataFrame) -> bytes:
    return dataframe_to_csv_text(df_full).encode("utf-8-sig")


def apply_status_updates(df_full: pd.DataFrame, edited_rows: pd.DataFrame) -> pd.DataFrame:
    df_new = df_full.copy()
    upd = edited_rows[["ROW_ID", "STATUS"]].copy()
    upd["STATUS"] = upd["STATUS"].apply(normalize_status)
    upd_map = dict(zip(upd["ROW_ID"], upd["STATUS"]))

    def _new_status(rid):
        if rid in upd_map:
            return upd_map[rid]
        return df_new.loc[df_new["ROW_ID"] == rid, "STATUS"].iloc[0]

    df_new["STATUS"] = df_new["ROW_ID"].map(_new_status)
    return df_new


def build_resumo_por_atleta(df_in: pd.DataFrame) -> pd.DataFrame:
    g = df_in.groupby("NOME DO ATLETA", dropna=False)
    resumo = g.agg(
        ocorrencias=("ROW_ID", "count"),
        ok=("STATUS", lambda s: (s == "Ok").sum()),
        nao_ok=("STATUS", lambda s: (s == "Não-Ok").sum()),
        pendente=("STATUS", lambda s: (s == "").sum()),
    ).reset_index()
    resumo["tem_pendencia"] = resumo["pendente"] > 0
    return resumo


# =========================
# Estado da sessão
# =========================
if "view" not in st.session_state:
    st.session_state.view = "lista"
if "athlete" not in st.session_state:
    st.session_state.athlete = ""
if "df_work" not in st.session_state:
    st.session_state.df_work = None
if "_gh_sha" not in st.session_state:
    st.session_state._gh_sha = ""
if "_gh_raw" not in st.session_state:
    st.session_state._gh_raw = ""


# =========================
# Carrega do GitHub
# =========================
cfg = get_cfg_from_secrets()

try:
    df_base, sha_base, raw_base = load_df_from_github(cfg)
except Exception as e:
    st.error(f"Erro ao carregar do GitHub: {e}")
    st.stop()

# Inicializa sessão com base do GitHub
if st.session_state.df_work is None:
    st.session_state.df_work = df_base.copy()
    st.session_state._gh_sha = sha_base
    st.session_state._gh_raw = raw_base

df = st.session_state.df_work


# =========================
# Sidebar - Filtros e ações
# =========================
st.sidebar.header("Fonte")
st.sidebar.write(f"Repo: `{cfg.repo}`")
st.sidebar.write(f"Branch: `{cfg.branch}`")
st.sidebar.write(f"Arquivo: `{cfg.csv_path}`")

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
colA, colB = st.sidebar.columns(2)

with colA:
    if st.button("Recarregar do GitHub", use_container_width=True):
        load_df_from_github.clear()
        df_base2, sha2, raw2 = load_df_from_github(cfg)
        st.session_state.df_work = df_base2.copy()
        st.session_state._gh_sha = sha2
        st.session_state._gh_raw = raw2
        st.session_state.view = "lista"
        st.session_state.athlete = ""
        st.rerun()

with colB:
    if st.button("Ir p/ Lista", use_container_width=True):
        st.session_state.view = "lista"
        st.rerun()

st.sidebar.caption("Pendente = STATUS vazio")


def apply_filters_for_list(df_in: pd.DataFrame) -> pd.DataFrame:
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


def save_to_github(current_df: pd.DataFrame, context_msg: str = "") -> None:
    """
    Salva o CSV atual no GitHub com commit.
    Faz 1 tentativa de retry se o sha estiver desatualizado.
    """
    new_text = dataframe_to_csv_text(current_df)

    # Evita commit se não mudou nada em relação ao último raw carregado
    if new_text.strip() == st.session_state._gh_raw.strip():
        st.info("Nenhuma alteração detectada para salvar.")
        return

    commit_msg = "Atualiza STATUS via app Streamlit"
    if context_msg:
        commit_msg += f" - {context_msg}"

    # 1) tenta com sha que temos
    try:
        gh_put_file(cfg, new_text=new_text, sha=st.session_state._gh_sha, message=commit_msg)
        st.success("Salvo no GitHub com commit ✅")
    except Exception as e1:
        # 2) retry: recarrega sha atual e tenta salvar novamente
        st.warning("O arquivo parece ter mudado no GitHub. Tentando salvar novamente com o SHA mais recente...")
        try:
            latest_raw, latest_sha = gh_get_file(cfg)
            gh_put_file(cfg, new_text=new_text, sha=latest_sha, message=commit_msg)
            st.success("Salvo no GitHub com commit ✅ (retry)")
        except Exception as e2:
            st.error(f"Falha ao salvar no GitHub. Detalhes:\n\n{e2}")
            return

    # Recarrega a base do GitHub para atualizar sha/raw e manter tudo consistente
    load_df_from_github.clear()
    df_base2, sha2, raw2 = load_df_from_github(cfg)
    st.session_state.df_work = df_base2.copy()
    st.session_state._gh_sha = sha2
    st.session_state._gh_raw = raw2
    st.session_state.view = "lista"
    st.rerun()


# =========================
# Views
# =========================
def view_lista():
    st.title("Transferências Internacionais — Atletas")

    df_f = apply_filters_for_list(df)

    resumo = build_resumo_por_atleta(df_f)
    if resumo.empty:
        st.info("Nenhum atleta encontrado com os filtros atuais.")
        st.download_button(
            "⬇️ Baixar CSV atualizado (sessão)",
            data=dataframe_to_csv_bytes(df),
            file_name="Transferencias_Internacionais_ATUALIZADO.csv",
            mime="text/csv",
            use_container_width=True,
        )
        return

    resumo = resumo.sort_values(["tem_pendencia", "ocorrencias", "NOME DO ATLETA"], ascending=[False, False, True])

    sem_pend = resumo[~resumo["tem_pendencia"]].copy()
    com_pend = resumo[resumo["tem_pendencia"]].copy()

    top1, top2, top3 = st.columns([1, 1, 2], vertical_alignment="center")
    with top1:
        if st.button("💾 Salvar no GitHub", type="primary", use_container_width=True):
            save_to_github(df, context_msg="lista")
    with top2:
        st.download_button(
            "⬇️ Baixar CSV (opcional)",
            data=dataframe_to_csv_bytes(df),
            file_name="Transferencias_Internacionais_ATUALIZADO.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with top3:
        st.caption("Dica: clique em “Salvar no GitHub” para gravar definitivamente sem baixar/subir arquivo.")

    col1, col2 = st.columns(2, vertical_alignment="top")

    with col1:
        st.subheader(f"✅ Atletas sem pendências ({len(sem_pend)})")
        st.dataframe(
            sem_pend.drop(columns=["tem_pendencia"]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Abrir ficha (sem pendências)**")
        atletas_sem = sem_pend["NOME DO ATLETA"].tolist()
        if atletas_sem:
            atleta_sel = st.selectbox("Selecione", atletas_sem, key="sel_sem")
            if st.button("Abrir (sem pendências)", use_container_width=True):
                st.session_state.athlete = atleta_sel
                st.session_state.view = "ficha"
                st.rerun()
        else:
            st.caption("Nenhum atleta sem pendências com os filtros atuais.")

    with col2:
        st.subheader(f"⚠️ Atletas com pendências ({len(com_pend)})")
        st.dataframe(
            com_pend.drop(columns=["tem_pendencia"]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Abrir ficha (com pendências)**")
        atletas_com = com_pend["NOME DO ATLETA"].tolist()
        if atletas_com:
            atleta_sel2 = st.selectbox("Selecione", atletas_com, key="sel_com")
            if st.button("Abrir (com pendências)", type="primary", use_container_width=True):
                st.session_state.athlete = atleta_sel2
                st.session_state.view = "ficha"
                st.rerun()
        else:
            st.caption("Nenhum atleta com pendências com os filtros atuais.")


def view_ficha():
    atleta = st.session_state.athlete
    if not atleta:
        st.session_state.view = "lista"
        st.rerun()

    st.title(f"Ficha do atleta: {atleta}")

    bar1, bar2, bar3 = st.columns([1, 1, 2], vertical_alignment="center")
    with bar1:
        if st.button("← Voltar", use_container_width=True):
            st.session_state.view = "lista"
            st.rerun()
    with bar2:
        if st.button("💾 Salvar no GitHub", type="primary", use_container_width=True):
            save_to_github(df, context_msg=f"atleta {atleta}")
    with bar3:
        st.caption("Ordenação: da data mais antiga para a mais recente. Salvar grava no GitHub com commit.")

    mask = df["NOME DO ATLETA"] == atleta
    df_a = df.loc[mask, ["ROW_ID", "DE", "PARA", "PAÍS", "DATA", "DATA_DT", "STATUS"]].copy()

    # Requisito: ordenar do mais antigo para o mais recente (NaT no fim)
    df_a = df_a.sort_values(["DATA_DT", "ROW_ID"], ascending=[True, True], na_position="last")
    df_show = df_a[["ROW_ID", "DE", "PARA", "PAÍS", "DATA", "STATUS"]].copy()

    st.caption(f"Ocorrências encontradas: **{len(df_show)}**")

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
        if st.button("Aplicar alterações (sessão)", type="secondary", use_container_width=True):
            st.session_state.df_work = apply_status_updates(df, edited)
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


# =========================
# Router
# =========================
if st.session_state.view == "lista":
    view_lista()
else:
    view_ficha()

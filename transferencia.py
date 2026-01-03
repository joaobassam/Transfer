import os
import pandas as pd
import streamlit as st

# =========================
# Configuração do App
# =========================
st.set_page_config(page_title="Transferências Internacionais", layout="wide")

CSV_PATH = "Transferencias_Internacionais_ATUALIZADO.csv"

STATUS_OPTIONS = ["", "Ok", "Não-Ok"]  # "" = pendente / sem marcação
STATUS_LABELS = {
    "": "(vazio) Pendente",
    "Ok": "Ok",
    "Não-Ok": "Não-Ok",
}


def normalize_status(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s if s in STATUS_OPTIONS else ""


@st.cache_data(show_spinner=False)
def load_data_from_local(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Arquivo CSV não encontrado: {path}\n"
            "Verifique se ele está no repositório e se o nome/caminho está correto."
        )

    df = pd.read_csv(path)

    required = ["NOME DO ATLETA", "DE", "PARA", "PAÍS", "DATA", "STATUS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV não contém colunas esperadas: {missing}")

    # Normaliza STATUS cedo (evita valores estranhos)
    df["STATUS"] = df["STATUS"].apply(normalize_status)

    # Datas / Ano
    df["DATA_DT"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
    df["ANO"] = df["DATA_DT"].dt.year

    # =========================
    # ID estável (SEM duplicar ROW_ID)
    # =========================
    # Caso comum: usuário baixou o CSV do app e subiu de volta (ROW_ID já existe)
    if "ROW_ID" in df.columns:
        # Se por acaso houver duplicidade de colunas, removemos depois
        # (mas ainda tentamos normalizar a coluna que sobrar)
        df["ROW_ID"] = pd.to_numeric(df["ROW_ID"], errors="coerce")

        # Se houver NaN, preenche com um id sequencial seguro
        if df["ROW_ID"].isna().any():
            df["ROW_ID"] = range(len(df))
    else:
        # Cria ROW_ID a partir do índice
        df = df.reset_index(drop=False).rename(columns={"index": "ROW_ID"})

    # Segurança extra: remove colunas duplicadas por nome (p.ex. ROW_ID repetido)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Garante que ROW_ID seja inteiro quando possível
    try:
        df["ROW_ID"] = pd.to_numeric(df["ROW_ID"], errors="coerce").fillna(range(len(df))).astype(int)
    except Exception:
        # fallback (não deve acontecer)
        df["ROW_ID"] = range(len(df))

    return df


def apply_status_updates(df_full: pd.DataFrame, edited_rows: pd.DataFrame) -> pd.DataFrame:
    df_new = df_full.copy()

    # Mantém somente o que importa
    upd = edited_rows[["ROW_ID", "STATUS"]].copy()
    upd["STATUS"] = upd["STATUS"].apply(normalize_status)
    upd["ROW_ID"] = pd.to_numeric(upd["ROW_ID"], errors="coerce")

    # Remove linhas sem ROW_ID válido
    upd = upd.dropna(subset=["ROW_ID"]).copy()
    upd["ROW_ID"] = upd["ROW_ID"].astype(int)

    upd_map = dict(zip(upd["ROW_ID"], upd["STATUS"]))

    # Atualiza por map (muito mais estável do que função linha a linha)
    df_new["STATUS"] = df_new["ROW_ID"].map(lambda rid: upd_map.get(int(rid), df_new.loc[df_new["ROW_ID"] == rid, "STATUS"].iloc[0]))

    return df_new


def dataframe_to_csv_bytes(df_full: pd.DataFrame) -> bytes:
    df_out = df_full.drop(columns=["DATA_DT", "ANO"], errors="ignore").copy()

    # Mantém ROW_ID no CSV (ID estável para reimportar)
    # Se você quiser remover ROW_ID do arquivo exportado, comente a linha abaixo:
    # df_out = df_out.drop(columns=["ROW_ID"], errors="ignore")

    return df_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def build_resumo_por_atleta(df_in: pd.DataFrame) -> pd.DataFrame:
    # Se o filtro deixar DF vazio, retorna vazio
    if df_in.empty:
        return pd.DataFrame(columns=["NOME DO ATLETA", "ocorrencias", "ok", "nao_ok", "pendente", "tem_pendencia"])

    # Garante que STATUS exista e esteja normalizado
    if "STATUS" not in df_in.columns:
        df_in = df_in.copy()
        df_in["STATUS"] = ""

    df_in = df_in.copy()
    df_in["STATUS"] = df_in["STATUS"].apply(normalize_status)

    # GroupBy
    g = df_in.groupby("NOME DO ATLETA", dropna=False)

    # Obs: 'count' em ROW_ID pode cair em bug se ROW_ID estiver duplicado como coluna.
    # A gente evita isso garantindo colunas não duplicadas no load_data_from_local.
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

# =========================
# Carrega dados
# =========================
try:
    df_base = load_data_from_local(CSV_PATH)
except Exception as e:
    st.error(f"Erro ao carregar CSV: {e}")
    st.stop()

if st.session_state.df_work is None:
    st.session_state.df_work = df_base.copy()

df = st.session_state.df_work

# =========================
# Sidebar - Filtros e Ações
# =========================
st.sidebar.header("Filtros")

anos_disponiveis = sorted([int(a) for a in df["ANO"].dropna().unique()]) if "ANO" in df.columns else []
ano_sel = st.sidebar.selectbox("Filtrar por ano (DATA)", options=["Todos"] + anos_disponiveis, index=0)

# Filtro por STATUS
status_sel = st.sidebar.multiselect(
    "Filtrar por STATUS",
    options=STATUS_OPTIONS,
    default=STATUS_OPTIONS,
    format_func=lambda s: STATUS_LABELS.get(s, str(s)),
)

busca = st.sidebar.text_input("Buscar atleta (contém)", value="").strip()

st.sidebar.divider()

colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("Recarregar CSV", use_container_width=True):
        # limpa cache pra pegar eventuais atualizações do arquivo no repo
        st.cache_data.clear()
        df_base = load_data_from_local(CSV_PATH)
        st.session_state.df_work = df_base.copy()
        st.session_state.view = "lista"
        st.session_state.athlete = ""
        st.rerun()

with colB:
    if st.button("Ir p/ Lista", use_container_width=True):
        st.session_state.view = "lista"
        st.rerun()

st.sidebar.caption("STATUS: vazio / Ok / Não-Ok")


def apply_filters_for_list(df_in: pd.DataFrame) -> pd.DataFrame:
    df_f = df_in.copy()

    if ano_sel != "Todos" and "ANO" in df_f.columns:
        df_f = df_f[df_f["ANO"] == int(ano_sel)]

    # status (se nada selecionado, não mostra nada)
    if status_sel:
        df_f = df_f[df_f["STATUS"].isin(status_sel)]
    else:
        df_f = df_f.iloc[0:0]

    # busca por nome
    if busca:
        df_f = df_f[df_f["NOME DO ATLETA"].str.contains(busca, case=False, na=False)]

    return df_f


# =========================
# Visão 1: Lista de atletas (duas listas)
# =========================
def view_lista():
    st.title("Transferências Internacionais — Atletas")

    df_f = apply_filters_for_list(df)

    resumo = build_resumo_por_atleta(df_f)
    if resumo.empty:
        st.info("Nenhum atleta encontrado com os filtros atuais.")
        st.subheader("Exportar")
        st.download_button(
            "⬇️ Baixar CSV atualizado",
            data=dataframe_to_csv_bytes(df),
            file_name="Transferencias_Internacionais_ATUALIZADO.csv",
            mime="text/csv",
            use_container_width=True,
        )
        return

    # Ordenação: mais ocorrências primeiro, depois nome
    resumo = resumo.sort_values(["ocorrencias", "NOME DO ATLETA"], ascending=[False, True])

    sem_pend = resumo[~resumo["tem_pendencia"]].copy()
    com_pend = resumo[resumo["tem_pendencia"]].copy()

    col1, col2 = st.columns(2, vertical_alignment="top")

    with col1:
        st.subheader(f"✅ Atletas sem pendências ({len(sem_pend)})")
        st.dataframe(
            sem_pend.drop(columns=["tem_pendencia"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "NOME DO ATLETA": st.column_config.TextColumn("Atleta"),
                "ocorrencias": st.column_config.NumberColumn("Ocorrências", format="%d"),
                "ok": st.column_config.NumberColumn("Ok", format="%d"),
                "nao_ok": st.column_config.NumberColumn("Não-Ok", format="%d"),
                "pendente": st.column_config.NumberColumn("Pendente", format="%d"),
            },
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
            column_config={
                "NOME DO ATLETA": st.column_config.TextColumn("Atleta"),
                "ocorrencias": st.column_config.NumberColumn("Ocorrências", format="%d"),
                "ok": st.column_config.NumberColumn("Ok", format="%d"),
                "nao_ok": st.column_config.NumberColumn("Não-Ok", format="%d"),
                "pendente": st.column_config.NumberColumn("Pendente", format="%d"),
            },
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

    st.divider()
    st.subheader("Exportar")
    st.download_button(
        "⬇️ Baixar CSV atualizado",
        data=dataframe_to_csv_bytes(df),
        file_name="Transferencias_Internacionais_ATUALIZADO.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================
# Visão 2: Ficha do atleta (ordenar por data antiga -> recente)
# =========================
def view_ficha():
    atleta = st.session_state.athlete
    if not atleta:
        st.session_state.view = "lista"
        st.rerun()

    st.title(f"Ficha do atleta: {atleta}")

    topA, topB = st.columns([1, 2], vertical_alignment="center")
    with topA:
        if st.button("← Voltar para lista", use_container_width=True):
            st.session_state.view = "lista"
            st.rerun()
    with topB:
        st.caption("Ordenação: da data mais antiga para a mais recente.")

    mask = df["NOME DO ATLETA"] == atleta
    df_a = df.loc[mask, ["ROW_ID", "DE", "PARA", "PAÍS", "DATA", "DATA_DT", "STATUS"]].copy()

    # Requisito 3: ordenar do mais antigo para o mais recente
    # (datas inválidas/NaT vão para o fim)
    df_a = df_a.sort_values(["DATA_DT", "ROW_ID"], ascending=[True, True], na_position="last")

    # Mostra sem DATA_DT
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
        if st.button("Aplicar alterações (nesta sessão)", type="primary", use_container_width=True):
            st.session_state.df_work = apply_status_updates(df, edited)
            st.success("Alterações aplicadas (em memória).")
            st.rerun()

    with c2:
        st.download_button(
            "⬇️ Baixar CSV atualizado",
            data=dataframe_to_csv_bytes(st.session_state.df_work),
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

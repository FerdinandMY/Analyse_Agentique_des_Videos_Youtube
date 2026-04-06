"""Reconstruit phase2_eda.ipynb avec les bons IDs et le bon contenu."""
import json, pathlib

def md(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

def code(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}

cells = [

md("c00", (
    "# Phase 2 \u2014 Exploratory Data Analysis (EDA)\n"
    "## YouTube Quality Analyzer \u2014 Corpus de commentaires\n\n"
    "**Objectif** : Explorer et valider le dataset CSV avant de lancer le pipeline multi-agents.\n\n"
    "**Livrables** :\n"
    "- Statistiques descriptives du corpus\n"
    "- Distribution des longueurs et des langues\n"
    "- M\u00e9triques d'engagement (author_likes, reply_count)\n"
    "- Flags de bruit heuristiques\n"
    "- Export vers `data/clean/`\n\n"
    "**Tag Git cible** : `v0.1.0`"
)),

md("c01", "---\n## 0. Configuration"),

code("c02", (
    "import random, re, html as html_lib\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "SEED = 42\n"
    "random.seed(SEED)\n"
    "np.random.seed(SEED)\n"
    "\n"
    "CSV_PATH     = '../data/raw/comments_raw.csv'\n"
    "CLEAN_OUTPUT = '../data/clean/comments_eda_preview.csv'\n"
    "\n"
    "REQUIRED_COLS = ['text']\n"
    "OPTIONAL_COLS = ['video_id', 'author_likes', 'reply_count']\n"
    "\n"
    "COLUMN_ALIASES = {\n"
    "    'texte_commentaire':    'text',\n"
    "    'comment_text':         'text',\n"
    "    'body':                 'text',\n"
    "    'nb_likes_commentaire': 'author_likes',\n"
    "    'likes':                'author_likes',\n"
    "    'nb_reponses':          'reply_count',\n"
    "    'replies':              'reply_count',\n"
    "    'commentaire_id':       'comment_id',\n"
    "    'publie_le':            'published_at',\n"
    "}\n"
    "\n"
    "print('Configuration OK \u2014 seed =', SEED)\n"
    "print('CSV source :', CSV_PATH)"
)),

md("c03", "---\n## 1. Chargement et validation du CSV"),

code("c04", (
    "df = pd.read_csv(CSV_PATH)\n"
    "print('Colonnes brutes du CSV :', list(df.columns))\n"
    "\n"
    "# Renommage automatique des colonnes (meme logique que A1 Loader)\n"
    "rename_map = {col: COLUMN_ALIASES[col] for col in df.columns if col in COLUMN_ALIASES}\n"
    "if rename_map:\n"
    "    df = df.rename(columns=rename_map)\n"
    "    print(f'Colonnes renommees : {rename_map}')\n"
    "\n"
    "print('Colonnes apres renommage :', list(df.columns))\n"
    "\n"
    "# Decodage HTML (&#39; -> ', <br> -> espace) -- present sur ~32% du corpus YouTube\n"
    "df['text'] = df['text'].astype(str).apply(\n"
    "    lambda t: re.sub(r'<[^>]+>', ' ', html_lib.unescape(t))\n"
    ")\n"
    "\n"
    "print(f'Shape : {df.shape[0]:,} lignes x {df.shape[1]} colonnes')\n"
    "df.head()"
)),

code("c05", (
    "missing_required = [c for c in REQUIRED_COLS if c not in df.columns]\n"
    "if missing_required:\n"
    "    raise ValueError(\n"
    "        f'Colonnes requises manquantes : {missing_required}\\n'\n"
    "        f'Colonnes disponibles : {list(df.columns)}'\n"
    "    )\n"
    "\n"
    "present_optional = [c for c in OPTIONAL_COLS if c in df.columns]\n"
    "missing_optional  = [c for c in OPTIONAL_COLS if c not in df.columns]\n"
    "print(f'OK Colonnes requises       : {REQUIRED_COLS}')\n"
    "print(f'OK Colonnes opt. presentes : {present_optional}')\n"
    "if missing_optional:\n"
    "    print(f'   Colonnes opt. absentes  : {missing_optional}')\n"
    "\n"
    "print(f'\\nValeurs nulles :')\n"
    "print(df[REQUIRED_COLS + present_optional].isnull().sum().to_string())\n"
    "print(f'\\nDoublons exacts (texte) : {df[\"text\"].duplicated().sum():,}')"
)),

code("c06", (
    "print('Types :')\n"
    "print(df.dtypes.to_string())"
)),

md("c07", "---\n## 2. Statistiques descriptives"),

code("c08", (
    "df_clean = df.copy()\n"
    "df_clean['text'] = df_clean['text'].astype(str).str.strip()\n"
    "df_clean = df_clean[df_clean['text'].str.len() > 0].copy()\n"
    "\n"
    "df_clean['char_length'] = df_clean['text'].str.len()\n"
    "df_clean['word_count']  = df_clean['text'].str.split().str.len()\n"
    "\n"
    "print(f'Apres suppression vides : {len(df_clean):,} (supprimes : {len(df) - len(df_clean)})')\n"
    "print()\n"
    "print(df_clean[['char_length', 'word_count']].describe().round(2).to_string())"
)),

code("c09", (
    "single_word = df_clean[df_clean['word_count'] == 1]\n"
    "print(f'Commentaires 1 mot : {len(single_word):,} ({100*len(single_word)/len(df_clean):.1f}%)')\n"
    "print(single_word['text'].value_counts().head(10).to_string())\n"
    "\n"
    "long_comments = df_clean[df_clean['word_count'] > 100]\n"
    "print(f'\\nCommentaires >100 mots : {len(long_comments):,} ({100*len(long_comments)/len(df_clean):.1f}%)')"
)),

md("c10", "---\n## 3. Distribution des longueurs"),

code("c11", (
    "import pathlib; pathlib.Path('../docs/phase2_data').mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "\n"
    "axes[0].hist(df_clean['char_length'].clip(upper=500), bins=50, color='steelblue', edgecolor='white')\n"
    "axes[0].set_title('Longueur en caracteres (cap 500)')\n"
    "axes[0].set_xlabel('Caracteres')\n"
    "axes[0].set_ylabel('Nb commentaires')\n"
    "axes[0].axvline(3, color='red', linestyle='--', label='Min A2 (3 chars)')\n"
    "axes[0].legend(fontsize=8)\n"
    "\n"
    "axes[1].hist(df_clean['word_count'].clip(upper=100), bins=50, color='teal', edgecolor='white')\n"
    "axes[1].set_title('Nombre de mots (cap 100)')\n"
    "axes[1].set_xlabel('Mots')\n"
    "axes[1].set_ylabel('Nb commentaires')\n"
    "axes[1].axvline(100, color='red', linestyle='--', label='Seuil long (Musleh 2023)')\n"
    "axes[1].legend(fontsize=8)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig('../docs/phase2_data/fig_length_distribution.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Sauvegarde : docs/phase2_data/fig_length_distribution.png')"
)),

md("c12", "---\n## 4. D\u00e9tection des langues"),

code("c13", (
    "def safe_detect(text):\n"
    "    try:\n"
    "        from langdetect import detect\n"
    "        return detect(str(text))\n"
    "    except Exception:\n"
    "        return 'unknown'\n"
    "\n"
    "sample_size = min(500, len(df_clean))\n"
    "df_sample = df_clean.sample(sample_size, random_state=SEED).copy()\n"
    "print(f'Detection de langue sur {sample_size} commentaires...')\n"
    "df_sample['language'] = df_sample['text'].apply(safe_detect)\n"
    "lang_counts = df_sample['language'].value_counts()\n"
    "print(lang_counts.head(10).to_string())"
)),

code("c14", (
    "fig, ax = plt.subplots(figsize=(10, 4))\n"
    "top_langs = lang_counts.head(10)\n"
    "bars = ax.barh(top_langs.index[::-1], top_langs.values[::-1], color='steelblue')\n"
    "ax.set_title('Top 10 langues detectees (500 commentaires)')\n"
    "ax.set_xlabel('Nb commentaires')\n"
    "for bar, val in zip(bars, top_langs.values[::-1]):\n"
    "    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,\n"
    "            f'{val} ({100*val/sample_size:.1f}%)', va='center', fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig('../docs/phase2_data/fig_language_distribution.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)),

md("c15", "---\n## 5. M\u00e9triques d'engagement"),

code("c16", (
    "engagement_cols = [c for c in ['author_likes', 'reply_count'] if c in df_clean.columns]\n"
    "\n"
    "if engagement_cols:\n"
    "    print(df_clean[engagement_cols].describe().round(2).to_string())\n"
    "\n"
    "    fig, axes = plt.subplots(1, len(engagement_cols), figsize=(6*len(engagement_cols), 4))\n"
    "    if len(engagement_cols) == 1:\n"
    "        axes = [axes]\n"
    "\n"
    "    for ax, col in zip(axes, engagement_cols):\n"
    "        cap = df_clean[col].quantile(0.95)\n"
    "        ax.hist(df_clean[col].dropna().clip(upper=cap), bins=40, color='coral', edgecolor='white')\n"
    "        ax.set_title(f'{col} (cap 95e percentile)')\n"
    "        ax.set_xlabel(col)\n"
    "        ax.set_ylabel('Nb commentaires')\n"
    "        med = df_clean[col].median()\n"
    "        ax.axvline(med, color='navy', linestyle='--', label=f'Mediane: {med:.0f}')\n"
    "        ax.legend(fontsize=9)\n"
    "\n"
    "    plt.tight_layout()\n"
    "    plt.savefig('../docs/phase2_data/fig_engagement_distribution.png', dpi=150, bbox_inches='tight')\n"
    "    plt.show()\n"
    "else:\n"
    "    print('Warning : author_likes et reply_count absents du CSV.')"
)),

md("c17", "---\n## 6. Analyse par vid\u00e9o"),

code("c18", (
    "if 'video_id' in df_clean.columns:\n"
    "    video_counts = df_clean['video_id'].value_counts()\n"
    "    print(f'Videos distinctes : {df_clean[\"video_id\"].nunique():,}')\n"
    "    print(f'Commentaires/video \u2014 moy : {video_counts.mean():.0f}, med : {video_counts.median():.0f}')\n"
    "    print(video_counts.head(10).to_frame('nb_commentaires').to_string())\n"
    "\n"
    "    fig, ax = plt.subplots(figsize=(10, 4))\n"
    "    ax.hist(video_counts, bins=30, color='mediumpurple', edgecolor='white')\n"
    "    ax.set_title('Distribution du nb de commentaires par video')\n"
    "    ax.set_xlabel('Nb commentaires')\n"
    "    ax.set_ylabel('Nb videos')\n"
    "    plt.tight_layout()\n"
    "    plt.savefig('../docs/phase2_data/fig_comments_per_video.png', dpi=150, bbox_inches='tight')\n"
    "    plt.show()\n"
    "else:\n"
    "    print('Warning : colonne video_id absente.')"
)),

md("c19", "---\n## 7. Heuristiques de bruit pr\u00e9-pipeline\n\nCes flags sont des estimations rapides **avant** le passage par A5 (Noise Detector LLM)."),

code("c20", (
    "def flag_noise(text):\n"
    "    text = str(text).strip()\n"
    "    if len(text) < 3: return 'trop_court'\n"
    "    if re.fullmatch(r'[^\\w\\s]+|[\\d\\W]+', text): return 'emoji_only'\n"
    "    if len(text.split()) == 1: return 'reaction_vide'\n"
    "    if re.search(r'https?://\\S+|www\\.\\S+|bit\\.ly/\\S+', text, re.IGNORECASE): return 'contient_url'\n"
    "    if re.search(r'(.)\\1{4,}', text): return 'lettres_repetees'\n"
    "    spam_kw = r'\\b(subscribe|abonne|abonnez|follow|check.?out|click.?here|gagne[rz]?|earn|bit\\.ly)\\b'\n"
    "    if re.search(spam_kw, text, re.IGNORECASE): return 'spam'\n"
    "    return 'ok'\n"
    "\n"
    "df_clean['noise_flag'] = df_clean['text'].apply(flag_noise)\n"
    "noise_dist = df_clean['noise_flag'].value_counts()\n"
    "print(noise_dist.to_string())\n"
    "ratio_bruit = 100 * (1 - noise_dist.get('ok', 0) / len(df_clean))\n"
    "print(f'\\nRatio de bruit estime (heuristique) : {ratio_bruit:.1f}%')"
)),

code("c21", (
    "fig, ax = plt.subplots(figsize=(9, 4))\n"
    "colors = ['#2ecc71' if k == 'ok' else '#e74c3c' for k in noise_dist.index]\n"
    "bars = ax.bar(noise_dist.index, noise_dist.values, color=colors, edgecolor='white')\n"
    "ax.set_title('Flags de bruit heuristique (pre-pipeline A5)')\n"
    "ax.set_ylabel('Nb commentaires')\n"
    "for bar, val in zip(bars, noise_dist.values):\n"
    "    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,\n"
    "            f'{val} ({100*val/len(df_clean):.1f}%)', ha='center', va='bottom', fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig('../docs/phase2_data/fig_noise_flags.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)),

md("c22", "---\n## 8. Exemples de commentaires"),

code("c23", (
    "clean_sample = df_clean[df_clean['noise_flag'] == 'ok'].copy()\n"
    "print(f'Commentaires propres : {len(clean_sample):,}')\n"
    "\n"
    "if 'author_likes' in clean_sample.columns:\n"
    "    print('\\nTop 5 les plus aimes :')\n"
    "    print(clean_sample.nlargest(5, 'author_likes')[['text', 'author_likes']].to_string())\n"
    "else:\n"
    "    print('\\n5 exemples aleatoires :')\n"
    "    print(clean_sample.sample(5, random_state=SEED)[['text']].to_string())"
)),

code("c24", (
    "noisy_sample = df_clean[df_clean['noise_flag'] != 'ok']\n"
    "if len(noisy_sample) > 0:\n"
    "    print(f'Exemples de commentaires bruites ({len(noisy_sample):,} total) :')\n"
    "    print(noisy_sample.groupby('noise_flag').head(2)[['noise_flag', 'text']].to_string())\n"
    "else:\n"
    "    print('Aucun commentaire noisy detecte.')"
)),

md("c25", "---\n## 9. R\u00e9sum\u00e9 et export"),

code("c26", (
    "n_total  = len(df)\n"
    "n_clean  = len(df_clean)\n"
    "n_ok     = len(df_clean[df_clean['noise_flag'] == 'ok'])\n"
    "n_noisy  = n_clean - n_ok\n"
    "n_dup    = df['text'].duplicated().sum()\n"
    "n_videos = df_clean['video_id'].nunique() if 'video_id' in df_clean.columns else 'N/A'\n"
    "\n"
    "print('=' * 52)\n"
    "print('RESUME EDA \u2014 CORPUS DE COMMENTAIRES')\n"
    "print('=' * 52)\n"
    "print(f'Total brut               : {n_total:>8,}')\n"
    "print(f'Doublons exacts          : {n_dup:>8,}  ({100*n_dup/n_total:.1f}%)')\n"
    "print(f'Apres filtrage A2        : {n_clean:>8,}')\n"
    "print(f'Propres (heuristique)    : {n_ok:>8,}  ({100*n_ok/n_clean:.1f}%)')\n"
    "print(f'Bruites (heuristique)    : {n_noisy:>8,}  ({100*n_noisy/n_clean:.1f}%)')\n"
    "print(f'Videos distinctes        : {str(n_videos):>8}')\n"
    "print(f'Longueur moy (chars)     : {df_clean[\"char_length\"].mean():>8.0f}')\n"
    "print(f'Longueur med (mots)      : {df_clean[\"word_count\"].median():>8.0f}')\n"
    "print('=' * 52)"
)),

code("c27", (
    "import pathlib\n"
    "pathlib.Path('../data/clean').mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "export_cols = ['text', 'noise_flag', 'char_length', 'word_count']\n"
    "for opt in ['video_id', 'author_likes', 'reply_count']:\n"
    "    if opt in df_clean.columns:\n"
    "        export_cols = [opt] + export_cols if opt == 'video_id' else export_cols + [opt]\n"
    "\n"
    "df_export = df_clean[[c for c in export_cols if c in df_clean.columns]]\n"
    "df_export.to_csv(CLEAN_OUTPUT, index=False, encoding='utf-8')\n"
    "print(f'Export : {CLEAN_OUTPUT} ({len(df_export):,} lignes)')"
)),

md("c28", (
    "---\n## 10. Gold Standard \u2014 Annotation automatique\n\n"
    "| \u00c9tape | M\u00e9thode |\n"
    "|---|---|\n"
    "| S\u00e9lection stratifi\u00e9e | 100 commentaires (strates : bruit \u00d7 longueur) |\n"
    "| Annotation | 2 passages LLM ind\u00e9pendants |\n"
    "| Kappa | Cohen's kappa \u2014 seuil 0.7 (AC-08 PRD) |\n"
    "| Export | `data/gold_standard/gold_standard.jsonl` |\n"
    "| Tag Git | `v0.1.0` si kappa > 0.7 |"
)),

code("c29", (
    "import subprocess, sys\n"
    "\n"
    "result = subprocess.run(\n"
    "    [\n"
    "        sys.executable,\n"
    "        '../scripts/annotate_gold_standard.py',\n"
    "        '--csv',       CSV_PATH,\n"
    "        '--n_samples', '100',\n"
    "        '--output',    '../data/gold_standard/gold_standard.jsonl',\n"
    "        # '--auto_tag',  # decommenter pour creer le tag Git v0.1.0\n"
    "    ],\n"
    "    capture_output=True, text=True,\n"
    ")\n"
    "print(result.stdout)\n"
    "if result.returncode != 0:\n"
    "    print('STDERR:', result.stderr[:800])"
)),

]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out = pathlib.Path("notebooks/phase2_eda.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook reconstruit : {len(cells)} cellules, {out.stat().st_size:,} octets.")

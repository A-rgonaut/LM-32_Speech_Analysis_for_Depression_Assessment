import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

def aggregate_sweep_results(results_base_dir="results/ssl"):
    """
    Scansiona le sottocartelle dei risultati seguendo il pattern 'ssl/<model_name>/layer<N>/',
    legge i file dev_results.csv e aggrega i dati in un unico DataFrame.
    """
    all_results = []
    
    p = Path(results_base_dir)
    if not p.exists():
        print(f"Errore: La directory dei risultati base '{results_base_dir}' non esiste.")
        return None

    # Scansiona tutte le sottocartelle di primo livello (i nomi dei modelli)
    for model_dir in p.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name_slug = model_dir.name # Es: 'facebook-wav2vec2-base-960h'
        
        # Scansiona tutte le sottocartelle di secondo livello (i layer)
        for layer_dir in model_dir.iterdir():
            if not layer_dir.is_dir() or not layer_dir.name.startswith('layer'):
                continue
                
            # Cerca il file dev_results.csv in questa specifica cartella
            result_file = layer_dir / 'dev_results.csv'
            
            if result_file.exists():
                try:
                    # Estrai il numero del layer dal nome della cartella
                    layer_match = re.search(r'\d+', layer_dir.name)
                    if not layer_match:
                        continue
                    layer = int(layer_match.group())

                    # Leggi il CSV
                    df_results = pd.read_csv(result_file, index_col='Metric')
                    
                    # Estrai il valore medio di f1_macro
                    f1_macro = df_results.loc['f1_macro', 'mean']
                    
                    all_results.append({
                        'model_name_slug': model_name_slug,
                        'layer': layer,
                        'dev_f1_score': f1_macro
                    })
                except (KeyError, IndexError, pd.errors.EmptyDataError) as e:
                    print(f"Attenzione: Impossibile processare il file {result_file}. Errore: {e}")
                    continue

    if not all_results:
        print(f"Nessun risultato valido trovato seguendo il pattern 'ssl/<model>/<layer>/' in '{results_base_dir}'.")
        print("Assicurati di aver eseguito prima lo script di test in modalità sweep.")
        return None
        
    print(f"Aggregati con successo i risultati da {len(all_results)} esperimenti.")
    return pd.DataFrame(all_results)

def plot_layer_performance():
    """
    Aggrega i risultati dello sweep e genera un grafico comparativo.
    """
    # --- 1. Aggrega i dati da tutti i file CSV ---
    df = aggregate_sweep_results()
    
    if df is None:
        return

    # --- 2. Preparazione dei dati per il plotting ---
    model_name_mapping = {
        'facebook-wav2vec2-base-960h': 'Wav2Vec2-Base',
        'microsoft-wavlm-base': 'WavLM-Base',
        'facebook-hubert-base-ls960': 'HuBERT-Base',
        'openai-whisper-small': 'Whisper-Small'
    }
    df['model_name'] = df['model_name_slug'].apply(lambda x: model_name_mapping.get(x, x))
    
    df = df.sort_values(by=['model_name', 'layer'])

    # --- 3. Creazione del Grafico ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.lineplot(
        data=df,
        x='layer',
        y='dev_f1_score',
        hue='model_name',
        marker='o',
        linestyle='-',
        ax=ax
    )

    # --- 4. Evidenziare il punto migliore per ogni modello ---
    best_points = df.loc[df.groupby('model_name')['dev_f1_score'].idxmax()]
    
    for _, row in best_points.iterrows():
        ax.plot(row['layer'], row['dev_f1_score'], '*', color='red', markersize=15, zorder=5)
        ax.annotate(f"{row['dev_f1_score']:.3f}",
                    xy=(row['layer'], row['dev_f1_score']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold')

    # --- 5. Abbellimento ---
    ax.set_title('Performance F1-Macro su Dev Set vs. Layer del Modello SSL', fontsize=16, fontweight='bold')
    ax.set_xlabel('Numero del Layer Transformer', fontsize=12)
    ax.set_ylabel('F1-Macro Score (Media su 5 Fold)', fontsize=12)
    ax.legend(title='Modello SSL', fontsize=10)
    
    max_layer = df['layer'].max()
    ax.set_xticks(range(0, max_layer + 1))
    ax.tick_params(axis='x')
    
    plt.tight_layout()

    # --- 6. Salvataggio ---
    output_path = "results/ssl/ssl_dev_set_performance_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nGrafico salvato con successo in: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_layer_performance()
import pandas as pd

def process_scores_ground_truth(input_file, output_file):
    """Processa un file CSV con le colonne 'secondi', 'tot_score_1', 'tot_score_2'
    calcolando le differenze tra i valori consecutivi delle colonne di punteggio.
    """
    # Leggi il file CSV senza intestazioni e nomina le colonne
    df = pd.read_csv(input_file, header=None, names=['secondi', 'tot_score_1', 'tot_score_2'])
    
    # Converti le colonne 'tot_score_1' e 'tot_score_2' in numeri (gestisce le conversioni da stringa a int)
    df['tot_score_1'] = pd.to_numeric(df['tot_score_1'], errors='coerce').fillna(0).astype(int)
    df['tot_score_2'] = pd.to_numeric(df['tot_score_2'], errors='coerce').fillna(0).astype(int)
    
    # Calcola le differenze per 'tot_score_1' e 'tot_score_2'
    df['tot_score_1'] = df['tot_score_1'].diff().fillna(0).astype(int)
    df['tot_score_2'] = df['tot_score_2'].diff().fillna(0).astype(int)
    
    # Scrivi il risultato su un nuovo file CSV
    df.to_csv(output_file, index=False, header=False)

    print(f"Operazione completata. I risultati sono stati scritti in '{output_file}'.")

def process_scores_inference(input_file, output_file):
    """Processa un file CSV con le colonne 'frame', 'bool1', 'bool2' aggregando i dati
    per gruppi di 30 frame e calcolando le somme per ciascun gruppo.
    """
    # Leggi il file CSV senza intestazioni e nomina le colonne
    df = pd.read_csv(input_file, header=None, names=['frame', 'bool1', 'bool2'])
    
    # Imposta la dimensione del collasso (30 frame alla volta)
    frame_size = 30
    
    # Aggiungi una colonna per il gruppo di frame
    df['group'] = df.index // frame_size
    
    # Calcola la somma per ciascun gruppo
    result = df.groupby('group').agg(
        sum_bool1=('bool1', 'sum'),  # Somma la seconda colonna (boolean)
        sum_bool2=('bool2', 'sum')   # Somma la terza colonna (boolean)
    ).reset_index(drop=True)
    
    # Aggiungi una colonna per il contatore dei secondi
    result['seconds'] = range(1, len(result) + 1)
    
    # Riorganizza le colonne per avere 'seconds' come prima
    result = result[['seconds', 'sum_bool1', 'sum_bool2']]
    
    # Scrivi il risultato su un nuovo file CSV
    result.to_csv(output_file, index=False)

    print(f"Operazione completata. I risultati sono stati scritti in '{output_file}'.")

def convert_seconds_to_hms(seconds):
    """Converte il numero di secondi in formato ore:minuti:secondi."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def validate_scores(ground_truth_file, inference_file):
    tp=0
    """Confronta i dati tra il file di inference e il file di ground truth.
    Verifica se nei 2 frame precedenti e successivi di ogni riga di inference c'è un '1' nella colonna 'tot_score_1' del ground truth.
    """
    ground_truth_df = pd.read_csv(ground_truth_file, header=None, names=['secondi', 'tot_score_1', 'tot_score_2'])
    inference_df = pd.read_csv(inference_file, header=None, names=['seconds', 'sum_bool1', 'sum_bool2'])
    
    # Converti le colonne 'sum_bool1' e 'sum_bool2' in numeri (gestisce le conversioni da stringa a int)
    ground_truth_df['tot_score_1'] = pd.to_numeric(ground_truth_df['tot_score_1'], errors='coerce').fillna(0).astype(int)
    ground_truth_df['tot_score_2'] = pd.to_numeric(ground_truth_df['tot_score_2'], errors='coerce').fillna(0).astype(int)
    inference_df['sum_bool1'] = pd.to_numeric(inference_df['sum_bool1'], errors='coerce').fillna(0).astype(int)
    inference_df['sum_bool2'] = pd.to_numeric(inference_df['sum_bool2'], errors='coerce').fillna(0).astype(int)
    
    for index, row in inference_df.iterrows():
        if row['sum_bool1'] > 0:
            second = row['seconds']
            # Trova la riga corrispondente in ground_truth (considerando la possibilità di +/- 2 righe)
            if second in ground_truth_df['secondi'].values:
                gt_index = ground_truth_df[ground_truth_df['secondi'] == second].index[0]
                start_index = max(0, gt_index - 3)
                end_index = min(len(ground_truth_df), gt_index + 4)
                if not any(ground_truth_df.loc[start_index:end_index, 'tot_score_1'] >0):
                    print(f"{convert_seconds_to_hms(int(second))}:  team1 false positive, tp = {tp}")
                else:
                    tp+=1
            else:
                print(f"Errore: Il secondo {second} non è presente nel file di ground truth.")

        if row['sum_bool2'] > 0:
            second = row['seconds']
            
            # Trova la riga corrispondente in ground_truth (considerando la possibilità di +/- 2 righe)
            if second in ground_truth_df['secondi'].values:
                gt_index = ground_truth_df[ground_truth_df['secondi'] == second].index[0]
                start_index = max(0, gt_index - 3)
                end_index = min(len(ground_truth_df), gt_index + 4)
                
                if not any(ground_truth_df.loc[start_index:end_index, 'tot_score_2'] > 0):
                     print(f"{convert_seconds_to_hms(int(second))}:  team2 false positive, tp = {tp}")
                else:
                    tp+=1
            else:
                print(f"Errore: Il secondo {second} non è presente nel file di ground truth.")


# Esempio di utilizzo
input_file_ground_truth = 'scores.csv'  
output_file_ground_truth = 'output_gorund.csv' 

input_file_inference = 'C:\\Users\\loren\\Downloads\\labels.txt'  
output_file_inference = 'output_inference.csv'  


process_scores_ground_truth(input_file_ground_truth, output_file_ground_truth)
process_scores_inference(input_file_inference, output_file_inference)
validate_scores(output_file_ground_truth, output_file_inference)

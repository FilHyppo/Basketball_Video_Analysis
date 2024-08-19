import yt_dlp

def scarica_video(youtube_url, output_path):
    # Configurazione delle opzioni di download
    ydl_opts = {
        'outtmpl': output_path,  # Percorso di salvataggio del file
        'format': 'best',  # Scarica il miglior formato video disponibile
    }

    # Scarica il video
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

# Esempio di utilizzo
youtube_url = 'https://www.youtube.com/watch?v=I7pTpMjqNRM'
output_path = "C:\\Users\\marco\\OneDrive\\Documents\\GitHub\\Basketball_Video_Analysis\\input_videos\\olympics.mp4"

scarica_video(youtube_url, output_path)

from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .forms import BasketballGameForm
from .models import BasketballGame
import cv2
import os
from . import utils, court_drawings, geometry
import json
import numpy as np
import random

def home_page(request):
    print(os.path.abspath(__file__))
    return render(request, 'court_detection/home.html')


def upload_video(request):
    if request.method == 'POST':
        form = BasketballGameForm(request.POST, request.FILES)
        if form.is_valid():
            game = form.save()
            fx, fy, cx, cy, k1, k2, k3, k4 = request.POST.get('fx'), request.POST.get('fy'), request.POST.get('cx'), request.POST.get('cy'), request.POST.get('k1'), request.POST.get('k2'), request.POST.get('k3'), request.POST.get('k4')
            game.distortion_parameters = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'k4': k4
            }
            game.save()
            return redirect('select_corners', game_id=game.id)
        else:
            print(form.errors)
    else:
        form = BasketballGameForm()
    return render(request, 'court_detection/upload_video.html', {'form': form})

def select_corners(request, game_id):
    game = get_object_or_404(BasketballGame, id=game_id)
    video_path = os.path.join(settings.MEDIA_ROOT, game.video.name)

    cap = cv2.VideoCapture(video_path)
    if settings.RANDOM_FRAME:
        # Estrai un frame in posizione random
        n = random.randint(0, cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        n = 0
    
    game.n = n
    game.save()
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    cap.release()


    #if not ret:
    #    return render(request, 'yourapp/error.html', {'message': 'Could not read video frame.'})

    # Salva il frame come immagine
    if game.distortion_parameters['fx']:
        frame = utils.undistort_frame(frame, utils.camera_matrix(game.distortion_parameters), utils.dist_coeffs(game.distortion_parameters))
    
    frame_path = os.path.join(settings.MEDIA_ROOT, 'frames', f'frame_{game_id}.jpg')
    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
    cv2.imwrite(frame_path, frame)

    frame_url = os.path.join(settings.MEDIA_URL, 'frames', f'frame_{game_id}.jpg')

    return render(request, 'court_detection/select_corners.html', {'game': game, 'frame_url': frame_url})

def new_frame(request, game_id):
    game = get_object_or_404(BasketballGame, id=game_id)
    video_path = game.video.path
    cap = cv2.VideoCapture(video_path)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, game.n + 5)
    ret, frame = cap.read()
    if game.distortion_parameters['fx']:
        frame = utils.undistort_frame(frame, utils.camera_matrix(game.distortion_parameters), utils.dist_coeffs(game.distortion_parameters))
    
    frame_path = os.path.join(settings.MEDIA_ROOT, 'frames', f'frame_{game_id}.jpg')
    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
    cv2.imwrite(frame_path, frame)
    game.n += 5
    game.save()
    return JsonResponse({'status': 'success'})
    


@csrf_exempt
def save_corners(request, game_id):
    if request.method == 'POST':
        game = get_object_or_404(BasketballGame, id=game_id)
        try:
            data = json.loads(request.body)
            corners = data.get('corners')
            if not corners or len(corners) < 4:
                return JsonResponse({'status': 'error', 'message': f'Invalid number of corners. Expected 6, got {len(corners)}'})
            
            cap = cv2.VideoCapture(game.video.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, game.n)
            ret, frame = cap.read()
            cap.release()
            h, w = frame.shape[:2]
            #Multiply each x corner by the width of the frame and each y corner by the height of the frame
            new_corners = dict()
            for corner in corners:
                new_corners[str(corner['id'])] =  {'x': int(corner['x'] * w), 'y': int(corner['y'] * h)} 
            
            new_corners = geometry.enhance_corners(frame, new_corners) #################################################################################

            game.corners = utils.find_missing_corners(frame, new_corners)
            game.save()
            
            return JsonResponse({'status': 'success'})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'})
        except Exception as e:
            print("Error:", str(e))
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def top_view(request, game_id):
    game = get_object_or_404(BasketballGame, id=game_id)
    output_path = os.path.join(settings.MEDIA_ROOT, 'top_views', f'top_view_{game_id}.jpg')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frame_path = os.path.join(settings.MEDIA_ROOT, 'frames', f'frame_{game_id}.jpg')
    frame = cv2.imread(frame_path)

    frame_top_view = utils.top_view(frame, game.corners)

    frame_top_view = cv2.resize(frame_top_view, (settings.TOP_VIEW_WIDTH, settings.TOP_VIEW_HEIGHT))

    court_drawings.draw_court_lines(frame_top_view)

    cv2.imwrite(output_path, frame_top_view)
    output_url = os.path.join(settings.MEDIA_URL, 'top_views', f'top_view_{game_id}.jpg')
    return render(request, 'court_detection/top_view.html', {'game': game, 'top_view_url': output_url})

def mask(request, game_id):
    game = get_object_or_404(BasketballGame, id=game_id)
    video_path = game.video.path
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    H, W = frame.shape[:2]

    top_view_path = os.path.join(settings.MEDIA_ROOT, 'top_views', f'top_view_{game_id}.jpg')
    top_view = cv2.imread(top_view_path)

    mask = utils.top_view(top_view, game.corners, inverse=True)

    output_path = os.path.join(settings.MEDIA_ROOT, 'masks', f'mask_{game_id}.jpg')
    cv2.imwrite(output_path, mask)
    mask_url = os.path.join(settings.MEDIA_URL, 'masks', f'mask_{game_id}.jpg')
    return render(request, 'court_detection/mask.html', {'game': game, 'mask_url': mask_url})

#TODO: CERCA DI STIMARE L'OPTICAL FLOW TRA I FRAME IN VIDEO IN CUI SI MUOVE LA TELECAMERA
def next_frame(request, game_id):
    game = get_object_or_404(BasketballGame, id=game_id)
    video_path = game.video.path
    n = game.n

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, base = cap.read()
    if game.distortion_parameters['fx']:
        base = utils.undistort_frame(base, utils.camera_matrix(game.distortion_parameters), utils.dist_coeffs(game.distortion_parameters))
    new_corners = game.corners
    output_image = base

    NUM_FRAMES = settings.NUM_FRAMES
    DIFF_BW_FRAMES = settings.DIFF_BW_FRAMES

    for i in range(1, NUM_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n + i*DIFF_BW_FRAMES)
        ret, frame = cap.read()
        if game.distortion_parameters['fx']:
            frame = utils.undistort_frame(frame, utils.camera_matrix(game.distortion_parameters), utils.dist_coeffs(game.distortion_parameters))
        if not ret:
            break
        output_image, new_corners = utils.landscape(output_image, frame, new_corners)
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, f'frame_{game_id}_{i}.jpg'), output_image)
    
    cap.release()


    court_drawings.draw_points(output_image, new_corners)
    #court_drawings.draw_lines(output_image, new_corners)

    top_output_image = utils.top_view(output_image, new_corners)

    output_path = os.path.join(settings.MEDIA_ROOT, 'next_frames', f'next_frame_{game_id}.jpg')
    cv2.imwrite(output_path, output_image)
    next_frame_url = os.path.join(settings.MEDIA_URL, 'next_frames', f'next_frame_{game_id}.jpg')
    top_output_path = os.path.join(settings.MEDIA_ROOT, 'next_frames', f'top_next_frame_{game_id}.jpg')
    cv2.imwrite(top_output_path, top_output_image)
    top_next_frame_url = os.path.join(settings.MEDIA_URL, 'next_frames', f'top_next_frame_{game_id}.jpg')
    return render(request, 'court_detection/next_frame.html', {'game': game, 
                                                               'next_frame_url': next_frame_url, 
                                                               'top_next_frame_url': top_next_frame_url})


def next_frame2(request, game_id, num_frames):
    game = get_object_or_404(BasketballGame, id=game_id)
    video_path = game.video.path
    game_n = game.n
    n = None
    prev_corners = None
    cap = None
    corners_positions = dict()

    # DIrectory in cui salvare i frame
    base_path = os.path.join(settings.MEDIA_ROOT, 'next_frames', f'game{game_id}')
    base_url = os.path.join(settings.MEDIA_URL, 'next_frames', f'game{game_id}')
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    # Frame già salvati
    file_names = os.listdir(base_path)
    file_names = [int(f.split('_')[-1].split('.')[0]) for f in file_names if f.startswith('frame_')]
    file_names = [f for f in file_names if f <= game_n + num_frames]
    if file_names:
        # Il numero più vicino e inferiore a game_n + num_frames
        n = max(file_names)

    # Prendo il frame più vicino a quello richiesto, ma solo se ci sono anche i suoi corners salvati
    if n is not None:
        base = cv2.imread(os.path.join(base_path, f'frame_{n}.jpg'))
        if os.path.exists(os.path.join(base_path, f'corners_positions.json')):
            with open(os.path.join(base_path, f'corners_positions.json'), 'r') as f:
                try:
                    corners_positions = json.load(f)
                    prev_corners = corners_positions[str(n)]
                    num_frames = game_n + num_frames - n
                    print("Trovato il frame già salvato:", n, ", me ne mancano ", num_frames)
                    print("Corners trovati:", prev_corners)
                except Exception as e:
                    prev_corners = None
                
    
    # Se non ho trovati i corner o prorpio non c'erano frame salvati parto da game_n (quello usato dall'utente per scegliere i corners)
    if prev_corners is None:
        print("Parto da game_n")
        n = game_n
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, game_n)
        ret, base = cap.read()
        if game.distortion_parameters['fx']:
            base = utils.undistort_frame(base, utils.camera_matrix(game.distortion_parameters), utils.dist_coeffs(game.distortion_parameters))
        prev_corners = game.corners     

    top_view = utils.top_view(base, prev_corners)
    
    prev = base
    corners_positions[n] = prev_corners
    cur = prev
    cur_corners = prev_corners

    DIFF_BW_FRAMES = settings.DIFF_BW_FRAMES
    stop = num_frames // DIFF_BW_FRAMES
    for i in range(1, stop + 1):
        frame_id = n + i*DIFF_BW_FRAMES
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, cur = cap.read()
        if game.distortion_parameters['fx']:
            cur = utils.undistort_frame(cur, utils.camera_matrix(game.distortion_parameters), utils.dist_coeffs(game.distortion_parameters))
        if not ret:
            break
        cur_corners = utils.new_corners(prev, cur, prev_corners)

        corners_positions[frame_id] = cur_corners

        copy = cur.copy()
        out_path = os.path.join(base_path, f'frame_{frame_id}.jpg')
        cv2.imwrite(out_path, copy)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        top_cur = utils.top_view(cur, cur_corners)
        mask = (top_cur.sum(axis=2) > 0).astype(np.uint8) * 255
        mask = cv2.bitwise_not(mask)
        top_view = cv2.bitwise_and(top_view, top_view, mask=mask)
        top_view = cv2.add(top_view, top_cur)

        prev_corners = cur_corners
        prev = cur


    if num_frames != stop * DIFF_BW_FRAMES:
        frame_id = n + num_frames
        if cap is None:
            cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, cur = cap.read()
        cur_corners = utils.new_corners(prev, cur, prev_corners)

        corners_positions[frame_id] = cur_corners

        copy = cur.copy()
        out_path = os.path.join(base_path, f'frame_{frame_id}.jpg')
        cv2.imwrite(out_path, copy)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)


        top_cur = utils.top_view(cur, cur_corners)
        mask = (top_cur.sum(axis=2) > 0).astype(np.uint8) * 255
        mask = cv2.bitwise_not(mask)
        top_view = cv2.bitwise_and(top_view, top_view, mask=mask)
        top_view = cv2.add(top_view, top_cur)

    if cap:
        cap.release()

    with open(os.path.join(base_path, f'corners_positions.json'), 'w') as f:
        json.dump(corners_positions, f)

    
    court_drawings.draw_points(cur, cur_corners)

    court_drawings.draw_court_lines(top_view)

    landscape = utils.top_view(top_view, cur_corners, inverse=True, new_h=2160, new_w=3840)

    output_path = os.path.join(base_path, f'next_frame_{game_id}.jpg')
    cv2.imwrite(output_path, cur)
    next_frame_url = os.path.join(base_url, f'next_frame_{game_id}.jpg')
    top_output_path = os.path.join(base_path, f'top_next_frame_{game_id}.jpg')
    cv2.imwrite(top_output_path, top_view)
    top_next_frame_url = os.path.join(base_url, f'top_next_frame_{game_id}.jpg')

    landscape_output_path = os.path.join(base_path, f'landscape_{game_id}.jpg')
    cv2.imwrite(landscape_output_path, landscape)
    landscape_url = os.path.join(base_url, f'landscape_{game_id}.jpg')

    return render(request, 'court_detection/next_frame.html', {'game': game, 
                                                               'next_frame_url': next_frame_url, 
                                                               'top_next_frame_url': top_next_frame_url,
                                                               'landscape_url': landscape_url})

#TODO: FAI Sì CHE utils.new_corners avverta se diminuire il numero di frame tra cur e prev
#TODO: CONTROLLARE SE IN /media/next_frames/ ci sono già frame salvati da cui partire




#TODO: per rilevare grossi cambiamenti usare l'istogramma colori
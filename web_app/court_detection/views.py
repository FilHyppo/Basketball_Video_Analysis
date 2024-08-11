from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .forms import BasketballGameForm
from .models import BasketballGame
import cv2
import os
from . import utils
import json
import numpy as np
import random

def home_page(request):
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

    # Estrai un frame in posizione random
    cap = cv2.VideoCapture(video_path)
    n = random.randint(0, cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

@csrf_exempt
def save_corners(request, game_id):
    if request.method == 'POST':
        game = get_object_or_404(BasketballGame, id=game_id)
        try:
            data = json.loads(request.body)
            corners = data.get('corners')
            print("corners", corners)
            if not corners or len(corners) < 4:
                return JsonResponse({'status': 'error', 'message': f'Invalid number of corners. Expected 6, got {len(corners)}'})
            
            cap = cv2.VideoCapture(game.video.path)
            ret, frame = cap.read()
            cap.release()
            h, w = frame.shape[:2]
            #Multiply each x corner by the width of the frame and each y corner by the height of the frame
            new_corners = dict()
            for corner in corners:
                new_corners[str(corner['id'])] =  {'x': int(corner['x'] * w), 'y': int(corner['y'] * h)} 
            
            game.corners = new_corners
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
    video_path = game.video.path
    output_path = os.path.join(settings.MEDIA_ROOT, 'top_views', f'top_view_{game_id}.jpg')
    output_path_undistort_frame = os.path.join(settings.MEDIA_ROOT, 'top_views', f'undistorted_{game_id}.jpg')
    output_path_distorted_frame = os.path.join(settings.MEDIA_ROOT, 'top_views', f'distorted_{game_id}.jpg')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frame_path = os.path.join(settings.MEDIA_ROOT, 'frames', f'frame_{game_id}.jpg')
    frame = cv2.imread(frame_path)

    #print("middle_top", middle_top)
    #print("middle_bottom", middle_bottom)
#
    #print("top_left", top_left)
    #print("bottom_right", bottom_right)
    #print("bottom_left", bottom_left)
    #print("top_right", top_right)

    undistort_frame = frame
    
    #utils.draw_points(undistort_frame, game.corners)
    cv2.imwrite(output_path_undistort_frame, undistort_frame)

    h, w = undistort_frame.shape[:2]
    h, w = h // 2, w // 2
    src = []
    dst = []
    points_names = []

    for id, corner in game.corners.items():
        src.append([corner['x'], corner['y']])
        dst.append([utils.corner_pos(id, w, h)])
        points_names.append(id)
        #print("Uso il punto", id, "con coordinate", corner)

    points = np.array(src, dtype=np.float32)
    points = points.reshape(-1, 1, 2)

    dst = np.array(dst, dtype=np.float32)
    dst = dst.reshape(-1, 1, 2)
    


    M, status = cv2.findHomography(points, dst)
    
    frame_top_view = cv2.warpPerspective(undistort_frame, M, (w, h))

    #utils.draw_lines(frame_top_view)

    #print("Salvo top view in: ", output_path)
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

    h, w = top_view.shape[:2]

    src = []
    dst = []
    points_names = []

    for id, corner in game.corners.items():
        dst.append([corner['x'], corner['y']])
        src.append([utils.corner_pos(id, w, h)])
        points_names.append(id)
        

    points = np.array(src, dtype=np.float32)
    points = points.reshape(-1, 1, 2)

    dst = np.array(dst, dtype=np.float32)
    dst = dst.reshape(-1, 1, 2)

    print("Punti sorgenti:", points)
    print("Punti destinazione:", dst)

    M, status = cv2.findHomography(points, dst)

    mask = cv2.warpPerspective(top_view, M, (W, H))

    output_path = os.path.join(settings.MEDIA_ROOT, 'masks', f'mask_{game_id}.jpg')
    cv2.imwrite(output_path, mask)
    mask_url = os.path.join(settings.MEDIA_URL, 'masks', f'mask_{game_id}.jpg')
    return render(request, 'court_detection/mask.html', {'game': game, 'mask_url': mask_url})